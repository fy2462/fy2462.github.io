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

    // 如果开启授权，添加授权中间件
	if serverConfig.Security.Secure {
		return serveGatewaySecure(ctx, pluginRegistry, serverConfig, authConfig.GetConfig(), storage.GetConfig(), additionalHandlers, adminScope)
	}

    // 启动非常规授权服务
	return serveGatewayInsecure(ctx, pluginRegistry, serverConfig, authConfig.GetConfig(), storage.GetConfig(), additionalHandlers, adminScope)
}
```
