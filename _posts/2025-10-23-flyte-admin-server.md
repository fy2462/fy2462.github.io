---
layout: post
title: 图解flyteML-flyte admin server
date: 2025-11-05 10:35:00.000000000 +08:00
---

> Flyte admin作为整个训练任务请求的中枢服务，对外承接WEB UI、SDK、CMD等工具链的通信服务，并与对象存储、DB联动，存储业务类数据，制定WF计划，编译WF模型成CRD，发送和收集第三方平台信息等。下面我们详细来看一下admin的业务流程。

## 1.总体架构

admin代码库中有两个入口，一个是server，另外一个是scheduler。
* server是启动grpc和http服务，用于接收外部请求。
* scheduler是启动另外的独立服务，从DB中fetch出LP相关任务，定时的去进行执行。

具体架构图如下:

![flyte_admin](/images/flyte/3/admin_server.png)

## 2.server详解

### 2.1插件注册

server函数入口为: flyteadmin/cmd/main.go, 首先创建插件注册模块

```go
func main() {
    glog.V(2).Info("Beginning Flyte Controller")
    err := entrypoints.Execute(plugins.NewRegistry())
    if err != nil {
        panic(err)
    }
}
```

admin将一些通用的端侧控制模块都一插件的方式进行组合，这种方式不仅方便开发者扩展和二次开发，并且代码上做了比较好的隔离，解耦能力强，通过配置可以进行集中的注册和配置, 预留的插件如下：

```go
const (
    // 扩展grpc endpoint
    PluginIDAdditionalGRPCService  PluginID = "AdditionalGRPCService"
    // 自定义授权header，需开启授权flag启用
    PluginIDCustomerHeaderMatcher  PluginID = "CustomerHeaderMatcher"
    // 内部已集成dataProxySvc，用于获取该服务
    PluginIDDataProxy              PluginID = "DataProxy"
    // 定义用户退出时的代码动作，此函数默认未注册
    PluginIDLogoutHook             PluginID = "LogoutHook"
    // 授权阶段重定向跳转前回调，允许返回错误时设置正确的HTTP状态码和提示信息，
    PluginIDPreRedirectHook        PluginID = "PreRedirectHook"
    // 请求ID拦截器, 认证全局授权
    PluginIDUnaryServiceMiddleware PluginID = "UnaryServiceMiddleware"
    // 用户WF的执行客户端，向目标集群发送CRD
    PluginIDWorkflowExecutor       PluginID = "WorkflowExecutor"
)
```

### 2.2服务设置与启动

admin使用cobra进行启动命令行参数的解析，在服务启动前会先初始化配置文件，从配置文件、环境变量、参数中读取字段并合并填充，构建全局配置文件供server进行使用。

服务目前支持如下命令参数

| 子命令 | 解释 |
|:-----|:-----|
|server| 启动admin server主进程 |
|secret init | 初始化secret provider |
|secret create | 用已存在的provider创建secret |
|migrate run | 执行所有迁移语句至DB |
|migrate rollback | 回滚某个迁移语句 |
|migrate seed-projects | 初始化默认项目 |
|clusterresource run| 运行定期同步任务，同步集群资源至DB |
|clusterresource run| 手动同步任务，同步集群资源至DB |

我们从server入手开始分析

```go
func Serve(ctx context.Context, pluginRegistry *plugins.Registry, additionalHandlers map[string]func(http.ResponseWriter, *http.Request)) error {
    // 获取服务配置
    serverConfig := config.GetConfig()
    configuration := runtime2.NewConfigurationProvider()
    adminScope := promutils.NewScope(configuration.ApplicationConfiguration().GetTopLevelConfig().GetMetricsScope()).NewSubScope("admin")

    // 如果开启授权，添加授权中间件，并添加证书认证
    if serverConfig.Security.Secure {
        return serveGatewaySecure(ctx, pluginRegistry, serverConfig, authConfig.GetConfig(), storage.GetConfig(), additionalHandlers, adminScope)
    }

    // 启动非常规授权服务
    return serveGatewayInsecure(ctx, pluginRegistry, serverConfig, authConfig.GetConfig(), storage.GetConfig(), additionalHandlers, adminScope)
}
```

我们以serveGatewaySecure为例进行讲解，进入函数后，首先是证书加载，和secret manager创建，用户加载secret

```go
storageCfg *storage.Config,
additionalHandlers map[string]func(http.ResponseWriter, *http.Request),
    scope promutils.Scope) error {
    certPool, cert, err := GetSslCredentials(ctx, cfg.Security.Ssl.CertificateFile, cfg.Security.Ssl.KeyFile)
    sm := secretmanager.NewFileEnvSecretManager(secretmanager.GetConfig())

    if err != nil {
        return err
    }
    // 如果开启授权，创建授权服务
    var authCtx interfaces.AuthenticationContext
    if cfg.Security.UseAuth {
        var oauth2Provider interfaces.OAuth2Provider
        var oauth2ResourceServer interfaces.OAuth2ResourceServer
        if authCfg.AppAuth.AuthServerType == authConfig.AuthorizationServerTypeSelf {
            oauth2Provider, err = authzserver.NewProvider(ctx, authCfg.AppAuth.SelfAuthServer, sm, scope.NewSubScope("auth_provider"))
            if err != nil {
                logger.Errorf(ctx, "Error creating authorization server %s", err)
                return err
            }

            oauth2ResourceServer = oauth2Provider
        } else {
            oauth2ResourceServer, err = authzserver.NewOAuth2ResourceServer(ctx, authCfg.AppAuth.ExternalAuthServer, authCfg.UserAuth.OpenID.BaseURL)
            if err != nil {
                logger.Errorf(ctx, "Error creating resource server %s", err)
                return err
            }
        }

        oauth2MetadataProvider := authzserver.NewService(authCfg)
        oidcUserInfoProvider := auth.NewUserInfoProvider()

        // 获得授权上下文，用于中间件注册
        authCtx, err = auth.NewAuthenticationContext(ctx, sm, oauth2Provider, oauth2ResourceServer, oauth2MetadataProvider, oidcUserInfoProvider, authCfg)
        if err != nil {
            logger.Errorf(ctx, "Error creating auth context %s", err)
            return err
        }
    }
```

### 2.3 GRPC服务设置

接下来创建GRPC server，这步是服务器核心

```go
// Creates a new gRPC Server with all the configuration
func newGRPCServer(ctx context.Context, pluginRegistry *plugins.Registry, cfg *config.ServerConfig,
    storageCfg *storage.Config, authCtx interfaces.AuthenticationContext,
    scope promutils.Scope, sm core.SecretManager, opts ...grpc.ServerOption) (*grpc.Server, error) {

    // 请求ID拦截器, 认证全局授权
    logger.Infof(ctx, "Registering default middleware with blanket auth validation")
    pluginRegistry.RegisterDefault(plugins.PluginIDUnaryServiceMiddleware, grpcmiddleware.ChainUnaryServer(
        RequestIDInterceptor, auth.BlanketAuthorization, auth.ExecutionUserIdentifierInterceptor))

    if cfg.GrpcConfig.EnableGrpcLatencyMetrics {
        logger.Debugf(ctx, "enabling grpc histogram metrics")
        grpcprometheus.EnableHandlingTimeHistogram()
    }

    // 针对流式请求注入中间件
    tracerProvider := otelutils.GetTracerProvider(otelutils.AdminServerTracer)
    otelUnaryServerInterceptor := otelgrpc.UnaryServerInterceptor(
        otelgrpc.WithTracerProvider(tracerProvider),
        otelgrpc.WithPropagators(propagation.TraceContext{}),
    )

    adminScope := scope.NewSubScope("admin")
    recoveryInterceptor := middleware.NewRecoveryInterceptor(adminScope)

    var chainedUnaryInterceptors grpc.UnaryServerInterceptor
    if cfg.Security.UseAuth {
        logger.Infof(ctx, "Creating gRPC server with authentication")
        // 添加所有中间件过滤器
        middlewareInterceptors := plugins.Get[grpc.UnaryServerInterceptor](pluginRegistry, plugins.PluginIDUnaryServiceMiddleware)
        chainedUnaryInterceptors = grpcmiddleware.ChainUnaryServer(
            // 处理panic和ote数据面采集中间件
            recoveryInterceptor.UnaryServerInterceptor(),
            grpcprometheus.UnaryServerInterceptor,
            otelUnaryServerInterceptor,
            // 授权相关中间件
            auth.GetAuthenticationCustomMetadataInterceptor(authCtx),
            grpcauth.UnaryServerInterceptor(auth.GetAuthenticationInterceptor(authCtx)),
            auth.AuthenticationLoggingInterceptor,
            middlewareInterceptors,
        )
    } else {
        logger.Infof(ctx, "Creating gRPC server without authentication")
        chainedUnaryInterceptors = grpcmiddleware.ChainUnaryServer(
            // 处理panic和ote数据面采集中间件
            recoveryInterceptor.UnaryServerInterceptor(),
            grpcprometheus.UnaryServerInterceptor,
            otelUnaryServerInterceptor,
        )
    }

    chainedStreamInterceptors := grpcmiddleware.ChainStreamServer(
        // 流式中间件
        recoveryInterceptor.StreamServerInterceptor(),
        grpcprometheus.StreamServerInterceptor,
    )

    // 注入拦截器
    serverOpts := []grpc.ServerOption{
        // 长连接请求拦截器
        grpc.StreamInterceptor(chainedStreamInterceptors),
        // 同步回复请求拦截器
        grpc.UnaryInterceptor(chainedUnaryInterceptors),
    }
    if cfg.GrpcConfig.MaxMessageSizeBytes > 0 {
        serverOpts = append(serverOpts, grpc.MaxRecvMsgSize(cfg.GrpcConfig.MaxMessageSizeBytes), grpc.MaxSendMsgSize(cfg.GrpcConfig.MaxMessageSizeBytes))
    }
    if cfg.GrpcConfig.MaxConcurrentStreams > 0 {
        serverOpts = append(serverOpts, grpc.MaxConcurrentStreams(uint32(cfg.GrpcConfig.MaxConcurrentStreams))) // #nosec G115
    }
    serverOpts = append(serverOpts, opts...)
    grpcServer := grpc.NewServer(serverOpts...)
    grpcprometheus.Register(grpcServer)
    dataStorageClient, err := storage.NewDataStore(storageCfg, scope.NewSubScope("storage"))
    if err != nil {
        logger.Error(ctx, "Failed to initialize storage config")
        panic(err)
    }

    configuration := runtime2.NewConfigurationProvider()
    // NewAdminServer是创建DB交互，关联gprc api的核心方法
    adminServer := adminservice.NewAdminServer(ctx, pluginRegistry, configuration, cfg.KubeConfig, cfg.Master, dataStorageClient, adminScope, sm)
    // 注册grpc path至adminServer
    grpcService.RegisterAdminServiceServer(grpcServer, adminServer)
    if cfg.Security.UseAuth {
        grpcService.RegisterAuthMetadataServiceServer(grpcServer, authCtx.AuthMetadataService())
        grpcService.RegisterIdentityServiceServer(grpcServer, authCtx.IdentityService())
    }
    // 数据网关服务，用于输入输出数据上传下载使用
    dataProxySvc, err := dataproxy.NewService(cfg.DataProxy, adminServer.NodeExecutionManager, dataStorageClient, adminServer.TaskExecutionManager)
    if err != nil {
        return nil, fmt.Errorf("failed to initialize dataProxy service. Error: %w", err)
    }
    // 向插件仓库注册中间件
    pluginRegistry.RegisterDefault(plugins.PluginIDDataProxy, dataProxySvc)
    grpcService.RegisterDataProxyServiceServer(grpcServer, plugins.Get[grpcService.DataProxyServiceServer](pluginRegistry, plugins.PluginIDDataProxy))

    grpcService.RegisterSignalServiceServer(grpcServer, rpc.NewSignalServer(ctx, configuration, scope.NewSubScope("signal")))

    // 为扩展插件预留
    additionalService := plugins.Get[common.RegisterAdditionalGRPCService](pluginRegistry, plugins.PluginIDAdditionalGRPCService)
    if additionalService != nil {
        if err := additionalService(ctx, grpcServer); err != nil {
            return nil, err
        }
    }

    // 注册grpc服务并返回
    healthServer := health.NewServer()
    healthServer.SetServingStatus("flyteadmin", grpc_health_v1.HealthCheckResponse_SERVING)
    grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
    if cfg.GrpcConfig.ServerReflection || cfg.GrpcServerReflection {
        reflection.Register(grpcServer)
    }

    return grpcServer, nil
}

```

下面对NewAdminServer进行解析，这里会创建各种业务逻辑的manager，并作为admin server成员变量hold使用。

1. create db and orm model
2. build execute cluster
3. create cluster wf engine
4. registry WorkflowExecutor plugin for cluster
5. create publisher+processor+event_publisher
6. create logic manager
7. create execution manager

```go
func NewAdminServer(ctx context.Context, pluginRegistry *plugins.Registry, configuration runtimeIfaces.Configuration,
    kubeConfig, master string, dataStorageClient *storage.DataStore, adminScope promutils.Scope, sm core.SecretManager) *AdminService {

    // ...
    // create some manager
    // ...

    logger.Info(ctx, "Initializing a new AdminService")
    return &AdminService{
        TaskManager: manager.NewTaskManager(repo, configuration, workflowengineImpl.NewCompiler(),
            adminScope.NewSubScope("task_manager")),
        WorkflowManager:          workflowManager,
        LaunchPlanManager:        launchPlanManager,
        ExecutionManager:         executionManager,
        NamedEntityManager:       namedEntityManager,
        DescriptionEntityManager: descriptionEntityManager,
        VersionManager:           versionManager,
        NodeExecutionManager:     nodeExecutionManager,
        TaskExecutionManager:     taskExecutionManager,
        ProjectManager:           manager.NewProjectManager(repo, configuration),
        ResourceManager:          resources.NewResourceManager(repo, configuration.ApplicationConfiguration()),
        MetricsManager: manager.NewMetricsManager(workflowManager, executionManager, nodeExecutionManager,
            taskExecutionManager, adminScope.NewSubScope("metrics_manager")),
        Metrics: InitMetrics(adminScope),
    }
    }
```

这其中个业务逻辑之间关联关系如架构图中所示

![flyte_admin](/images/flyte/3/admin_server_managers.png)

总结来看，关系中分为3个层次: 

* 执行层: 用于定时任务、node、task级别执行是的统一管理入口
* 数据层: 用户执行后的数据流动，与其他子系统交互，和相关事件接收处理
* 逻辑层: 用户实现具体的grpc api逻辑，和内部可观测管理等。

### 2.3 HTTP服务设置和关联

接下来看一下newHTTPServer，我们看一下http服务是如何创建的，http服务也是根据proto中定义创建rest接口，并关联注册grpc服务，其核心代码如下:

```go
func newHTTPServer(ctx context.Context, pluginRegistry *plugins.Registry, cfg *config.ServerConfig, _ *authConfig.Config, authCtx interfaces.AuthenticationContext,
    additionalHandlers map[string]func(http.ResponseWriter, *http.Request),
    grpcAddress string, grpcConnectionOpts ...grpc.DialOption) (*http.ServeMux, error) {
    // 做http请求和grpc server关联，将http请求转换成grpc request发送给grpc监听地址
    err := service.RegisterAdminServiceHandlerFromEndpoint(ctx, gwmux, grpcAddress, grpcConnectionOpts)
    if err != nil {
        return nil, errors.Wrap(err, "error registering admin service")
    }

    err = service.RegisterAuthMetadataServiceHandlerFromEndpoint(ctx, gwmux, grpcAddress, grpcConnectionOpts)
    if err != nil {
        return nil, errors.Wrap(err, "error registering auth service")
    }

    err = service.RegisterIdentityServiceHandlerFromEndpoint(ctx, gwmux, grpcAddress, grpcConnectionOpts)
    if err != nil {
        return nil, errors.Wrap(err, "error registering identity service")
    }

    err = service.RegisterDataProxyServiceHandlerFromEndpoint(ctx, gwmux, grpcAddress, grpcConnectionOpts)
    if err != nil {
        return nil, errors.Wrap(err, "error registering data proxy service")
    }

    err = service.RegisterSignalServiceHandlerFromEndpoint(ctx, gwmux, grpcAddress, grpcConnectionOpts)
    if err != nil {
        return nil, errors.Wrap(err, "error registering signal service")
    }

    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        ctx := GetOrGenerateRequestIDForRequest(r)
        gwmux.ServeHTTP(w, r.WithContext(ctx))
    })
    }
```

对于data proxy grpc来说，由于需要上传和下载文件数据，需要通过node execution manager 和 task execution manager获取相关任务的基础信息，并存储和生成uri写入至db中。

auth grpc入口是metadata_provider.go，主要提供auth2的meta数据，可以从本地服务生成，也可以从第三方服务获取，逻辑较为简单，就不深入展开。

![flyte_admin](/images/flyte/3/admin_auth_and_data_grpc.png)

## 3. WF业务流程举例

接下来，我们以一个实际场景的调用说明一下整个流程如何进行的，先给出一个关系图
我们从上节文章了解到，flytekit在发送任务前会执行2个步骤: 1.registry_script 2.execute_entity
请求发送至admin后，对应如下几个接口

![flyte_admin](/images/flyte/3/admin_wf_execution.png)

### 3.1 注册entity

其中TaskManger和LaunchPlanManger主要是对entity做本地检查和转换，并在DB中记录相关请求，同时也会以LP->WF->TASK等依赖关系补全执行链

注册阶段我们重点说一下createWorkfolw:

1. 检查是否存储以后的wf
2. 收集关联task并创建task map
3. 创建PL provider(本地或者第三方云厂商接口)
4. 编译WF

编译阶段会根据从请求中拿到的template递归编译成DAG，并验证DAG是否有环，出度入度是否正确等，并将dag存入数据库中

```go
// GetRequirements computes requirements for a given Workflow.
func GetRequirements(fg *core.WorkflowTemplate, subWfs []*core.WorkflowTemplate) (reqs WorkflowExecutionRequirements, err error) {
    errs := errors.NewCompileErrors()
    compiledSubWfs := toCompiledWorkflows(subWfs...)

    index, ok := common.NewWorkflowIndex(compiledSubWfs, errs)

    if ok {
        return getRequirements(fg, index, true, errs), nil
    }

    return WorkflowExecutionRequirements{}, errs
}
```

### 3.2 执行entity

执行阶段:

1. 会根据请求发送到类型判断并选择执行PL/WF流程，还是单Task流程，其中主要区别点是task会关联LP信息，PL/WF会查询验证是否注册过了
2. 如何WF存在大引用对象(闭包)，从对象存储中获取
3. 保存任务输入和用户输入到对象存储
4. 设置任务执行资源和使用哪个queue
5. 构建执行参数
6. 创建模型和CRD，将CRD发送至执行服务器(propeller)
7. 写入数据库，然后返回response

## 4. 总结

本节重点梳理了admin服务的整体架构图和依赖关系，通过wf注册和运行为切入点，梳理的执行力流程。下一节讲介绍flyte的执行器propeller，执行器会负责具体k8s层的任务调度和状态监控、反馈等环节。
