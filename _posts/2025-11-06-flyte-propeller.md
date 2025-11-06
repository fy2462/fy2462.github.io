---
layout: post
title: 图解flyteML-propeller
date: 2025-11-06 10:35:00.000000000 +08:00
---

> Flyte propeller作为集群当中真正的执行器，负责CRD的监听、集群资源的分配、三方子系统的交互和向admin server反馈等等

## 1.总体架构

propeller命令行有3个子命令，总体架构图如下

* init-certs：用于生成webhook访问api-server的证书
* webhook：用于向监听pod中注入注入环境变量，为运行flyte SDK和访问api-server提供secret
* controller-server: 用于接收admin发来的CRD，创建controller从插件仓库中生成指定模板，然后创建对应pod或者CRD，发送给k8s或子系统。

![flyte_propeller](/images/flyte/4/flyte_propeller_overview.png)

我们还是从先从controller开始讲起

## 2. 创建controller

controller启动的入口函数是executeRootCmd

```go

func executeRootCmd(baseCtx context.Context, cfg *config2.Config) error {
    // 处理退出信号
    ctx := signals.SetupSignalHandler(baseCtx)

    // 设置prometheus labels
    keys := contextutils.MetricKeysFromStrings(cfg.MetricKeys)
    logger.Infof(context.TODO(), "setting metrics keys to %+v", keys)
    if len(keys) > 0 {
        labeled.SetMetricKeys(keys...)
    }

    // 注册服务追踪
    for _, serviceName := range []string{otelutils.AdminClientTracer, otelutils.BlobstoreClientTracer,
        otelutils.DataCatalogClientTracer, otelutils.FlytePropellerTracer, otelutils.K8sClientTracer} {
        if err := otelutils.RegisterTracerProviderWithContext(ctx, serviceName, otelutils.GetConfig()); err != nil {
            logger.Errorf(ctx, "Failed to create otel tracer provider. %v", err)
            return err
        }
    }

    // Add the propeller subscope because the MetricsPrefix only has "flyte:" to get uniform collection of metrics.
    propellerScope := promutils.NewScope(cfg.MetricsPrefix).NewSubScope("propeller").NewSubScope(cfg.LimitNamespace)
    limitNamespace := ""
    var namespaceConfigs map[string]cache.Config
    if cfg.LimitNamespace != defaultNamespace {
        limitNamespace = cfg.LimitNamespace
        namespaceConfigs = map[string]cache.Config{
            limitNamespace: {},
        }
    }

    options := manager.Options{
        Cache: cache.Options{
            SyncPeriod:        &cfg.DownstreamEval.Duration,
            DefaultNamespaces: namespaceConfigs,
        },
        NewCache:  executors.NewCache,
        NewClient: executors.BuildNewClientFunc(propellerScope),
        Metrics: metricsserver.Options{
            // Disable metrics serving
            BindAddress: "0",
        },
    }

    // 向api server注册controller，k8s 会创建informer、client、cache、leaderelection对应的资源
    /**
    ┌────────────────────────────┐
    │     Controller Manager     │
    │      (control center)      │
    │                            │
    │  ┌──────────────────────┐  │
    │  │     Controllers      │  │
    │  │     (Reconciler)     │  │
    │  └─────────┬────────────┘  │
    │            │               │
    │       uses Client          │
    │            │               │
    │       uses Cache           |
    |        (Informer)          │
    │            │               │
    │            ▼               │
    │        API Server          │
    │                            │
    │ Webhook Server(admission)  │
    └────────────────────────────┘
     **/
    // API Request -> Authentication & Authorization -> Admission（Webhook) ->
    // Persist to etcd -> Reconciliation (controller) -> Status Update
    mgr, err := controller.CreateControllerManager(ctx, cfg, options)
    if err != nil {
        logger.Fatalf(ctx, "Failed to create controller manager. Error: %v", err)
        return err
    }

    handlers := map[string]http.Handler{
        "/k8smetrics": promhttp.HandlerFor(metrics.Registry,
            promhttp.HandlerOpts{
                ErrorHandling: promhttp.HTTPErrorOnError,
            },
        ),
    }

    g, childCtx := errgroup.WithContext(ctx)
    g.Go(func() error {
        err := profutils.StartProfilingServerWithDefaultHandlers(childCtx, cfg.ProfilerPort.Port, handlers)
        if err != nil {
            logger.Fatalf(childCtx, "Failed to Start profiling and metrics server. Error: %v", err)
        }
        return err
    })

    // 1. 设置性能采样label
    // 2. 启动controller manager
    g.Go(func() error {
        err := controller.StartControllerManager(childCtx, mgr)
        if err != nil {
            logger.Fatalf(childCtx, "Failed to start controller manager. Error: %v", err)
        }
        return err
    })

    g.Go(func() error {
        // controller 核心函数
        err := controller.StartController(childCtx, cfg, defaultNamespace, mgr, &propellerScope)
        if err != nil {
            logger.Fatalf(childCtx, "Failed to start controller. Error: %v", err)
        }
        return err
    })

    return g.Wait()
}

```

最终会启动异步协程运行controller.StartController，我们再看一下这个代码

```go
func StartController(ctx context.Context, cfg *config.Config, defaultNamespace string, mgr manager.Manager, scope *promutils.Scope) error {
    // Setup cancel on the context
    ctx, cancel := context.WithCancel(ctx)
    defer cancel()

    kubeClient, kubecfg, err := utils.GetKubeConfig(ctx, cfg)
    if err != nil {
        return errors.Wrapf(err, "error building Kubernetes Clientset")
    }

    // k8sResolver 主要解析k8s 中的 service
    // resolver监听目标service的ip、port变化后, 更新
    resolver.Register(k8sResolver.NewBuilder(ctx, kubeClient, k8sResolver.Schema))

    flyteworkflowClient, err := clientset.NewForConfig(kubecfg)
    if err != nil {
        return errors.Wrapf(err, "error building FlyteWorkflow clientset")
    }

    // 如何CRD没有注册，创建FlyteWorkflow CRD
    if cfg.CreateFlyteWorkflowCRD {
        logger.Infof(ctx, "creating FlyteWorkflow CRD")
        apiextensionsClient, err := apiextensionsclientset.NewForConfig(kubecfg)
        if err != nil {
            return errors.Wrapf(err, "error building apiextensions clientset")
        }

        _, err = apiextensionsClient.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, &flyteworkflow.CRD, v1.CreateOptions{})
        if err != nil {
            if apierrors.IsAlreadyExists(err) {
                logger.Warnf(ctx, "FlyteWorkflow CRD already exists")
            } else {
                return errors.Wrapf(err, "failed to create FlyteWorkflow CRD")
            }
        }
    }

    opts := SharedInformerOptions(cfg, defaultNamespace)
    flyteworkflowInformerFactory := informers.NewSharedInformerFactoryWithOptions(flyteworkflowClient, cfg.WorkflowReEval.Duration, opts...)

    informerFactory := k8sInformers.NewSharedInformerFactoryWithOptions(kubeClient, flyteK8sConfig.GetK8sPluginConfig().DefaultPodTemplateResync.Duration)

    // 1. 创建admin grpc客户端
    // 2. 创建metadata s3 store
    // 3. 创建其他交互系统客户端封装: admin client
    // 4. 创建GC，定制回收CRD目标的namespace资源
    // 5. 创建CRD information
    // 6. 创建 catalogClient
    // 7. 创建请求队列
    // 8. 创建业务监控和LP status
    // 9. 创建node、wf的执行器
    // 10. 关联inform的回调函数
    c, err := New(ctx, cfg, kubeClient, flyteworkflowClient, flyteworkflowInformerFactory, informerFactory, mgr, *scope)
    if err != nil {
        return errors.Wrap(err, "failed to start FlytePropeller")
    } else if c == nil {
        return errors.Errorf("Failed to create a new instance of FlytePropeller")
    }

    // 观测CRD变更
    go flyteworkflowInformerFactory.Start(ctx.Done())
    // 观测k8s变化
    go informerFactory.Start(ctx.Done())

    // 1.运行Controller、启动线程池、监控、GC检查等
    if err = c.Run(ctx); err != nil {
        return errors.Wrapf(err, "Error running FlytePropeller.")
    }
    return nil
}

```

可以看到New函数是真正创建服务组件的核心函数，

```go
func New(ctx context.Context, cfg *config.Config, kubeClientset kubernetes.Interface, flytepropellerClientset clientset.Interface,
    flyteworkflowInformerFactory informers.SharedInformerFactory, informerFactory k8sInformers.SharedInformerFactory,
    kubeClient executors.Client, scope promutils.Scope) (*Controller, error) {

    // 创建admin server client
    adminClient, signalClient, authOpts, err := getAdminClient(ctx)
    if err != nil {
        logger.Errorf(ctx, "failed to initialize Admin client, err :%s", err.Error())
        return nil, err
    }

    sCfg := storage.GetConfig()
    if sCfg == nil {
        logger.Errorf(ctx, "Storage configuration missing.")
    }
    store, err := storage.NewDataStore(sCfg, scope.NewSubScope("metastore"))
    if err != nil {
        return nil, errors.Wrapf(err, "Failed to create Metadata storage")
    }

    logger.Info(ctx, "Setting up event sink and recorder")
    eventSink, err := events.ConstructEventSink(ctx, events.GetConfig(ctx), scope.NewSubScope("event_sink"))
    if err != nil {
        return nil, errors.Wrapf(err, "Failed to create EventSink [%v], error %v", events.GetConfig(ctx).Type, err)
    }
    gc, err := NewGarbageCollector(cfg, scope, clock.RealClock{}, kubeClientset.CoreV1().Namespaces(), flytepropellerClientset.FlyteworkflowV1alpha1())
    if err != nil {
        logger.Errorf(ctx, "failed to initialize GC for workflows")
        return nil, errors.Wrapf(err, "failed to initialize WF GC")
    }

    eventRecorder, err := utils.NewK8sEventRecorder(ctx, kubeClientset, controllerAgentName, cfg.PublishK8sEvents)
    if err != nil {
        logger.Errorf(ctx, "failed to event recorder %v", err)
        return nil, errors.Wrapf(err, "failed to initialize resource lock.")
    }
    controller := &Controller{
        metrics:    newControllerMetrics(scope),
        recorder:   eventRecorder,
        gc:         gc,
        numWorkers: cfg.Workers,
    }

    // 尝试获取master lock
    lock, err := leader.NewResourceLock(kubeClientset.CoreV1(), kubeClientset.CoordinationV1(), eventRecorder, cfg.LeaderElection)
    if err != nil {
        logger.Errorf(ctx, "failed to initialize resource lock.")
        return nil, errors.Wrapf(err, "failed to initialize resource lock.")
    }

    if lock != nil {
        logger.Infof(ctx, "Creating leader elector for the controller.")
        controller.leaderElector, err = leader.NewLeaderElector(lock, cfg.LeaderElection, controller.onStartedLeading, func() {
            logger.Fatal(ctx, "Lost leader state. Shutting down.")
        })

        if err != nil {
            logger.Errorf(ctx, "failed to initialize leader elector.")
            return nil, errors.Wrapf(err, "failed to initialize leader elector.")
        }
    }

    // 创建 client/informers/externalversions/flyteworkflow/v1alpha1 监听
    flyteworkflowInformer := flyteworkflowInformerFactory.Flyteworkflow().V1alpha1().FlyteWorkflows()
    controller.flyteworkflowSynced = flyteworkflowInformer.Informer().HasSynced

    podTemplateInformer := informerFactory.Core().V1().PodTemplates()

    // set default namespace(flyte) for pod template store
    podNamespace, found := os.LookupEnv(podNamespaceEnvVar)
    if !found {
        podNamespace = podDefaultNamespace
    }

    // pod创建的默认命名空间: flyte
    flytek8s.DefaultPodTemplateStore.SetDefaultNamespace(podNamespace)

    logger.Info(ctx, "Setting up Catalog client.")
    catalogClient, err := catalog.NewCatalogClient(ctx, authOpts...)
    if err != nil {
        return nil, errors.Wrapf(err, "Failed to create datacatalog client")
    }

    workQ, err := NewCompositeWorkQueue(ctx, cfg.Queue, scope)
    if err != nil {
        return nil, errors.Wrapf(err, "Failed to create WorkQueue [%v]", scope.CurrentScope())
    }
    controller.workQueue = workQ

    // 内存中设置WF保存中间件，可以按workflowStore policy，指定缓存的颗粒度
    controller.workflowStore, err = workflowstore.NewWorkflowStore(ctx, workflowstore.GetConfig(), flyteworkflowInformer.Lister(), flytepropellerClientset.FlyteworkflowV1alpha1(), scope)
    if err != nil {
        return nil, stdErrs.Wrapf(errors3.CausedByError, err, "failed to initialize workflow store")
    }

    // 用于记录业务处理耗费的时间
    controller.levelMonitor = NewResourceLevelMonitor(scope.NewSubScope("collector"), flyteworkflowInformer.Lister())

    // 为launchplan的同步模块，向admin同步LP状态，可以自动进行数据同步
    var launchPlanActor launchplan.FlyteAdmin
    if cfg.EnableAdminLauncher {
        launchPlanActor, err = launchplan.NewAdminLaunchPlanExecutor(ctx, adminClient, launchplan.GetAdminConfig(),
            scope.NewSubScope("admin_launcher"), store, controller.enqueueWorkflowForNodeUpdates)
        if err != nil {
            logger.Errorf(ctx, "failed to create Admin workflow Launcher, err: %v", err.Error())
            return nil, err
        }

        if err := launchPlanActor.Initialize(ctx); err != nil {
            logger.Errorf(ctx, "failed to initialize Admin workflow Launcher, err: %v", err.Error())
            return nil, err
        }
    } else {
        launchPlanActor = launchplan.NewFailFastLaunchPlanExecutor()
    }

    recoveryClient := recovery.NewClient(adminClient)
    // 对DAG图中的task、branch、workflow、gate、array、start、end类型，注册handle方法
    nodeHandlerFactory, err := factory.NewHandlerFactory(ctx, launchPlanActor, launchPlanActor,
        kubeClient, kubeClientset, catalogClient, recoveryClient, &cfg.EventConfig, cfg.LiteralOffloadingConfig, cfg.ClusterID, signalClient, scope)
    if err != nil {
        return nil, errors.Wrapf(err, "failed to create node handler factory")
    }

    // 创建node executor，用户DAG的执行
    nodeExecutor, err := nodes.NewExecutor(ctx, cfg.NodeConfig, store, controller.enqueueWorkflowForNodeUpdates, eventSink,
        launchPlanActor, launchPlanActor, storage.DataReference(cfg.DefaultRawOutputPrefix), kubeClient,
        catalogClient, recoveryClient, cfg.LiteralOffloadingConfig, &cfg.EventConfig, cfg.ClusterID, signalClient, nodeHandlerFactory, scope)
    if err != nil {
        return nil, errors.Wrapf(err, "Failed to create Controller.")
    }

    activeExecutions, err := workflowstore.NewExecutionStatsHolder()
    if err != nil {
        return nil, err
    }
    // 对执行的时间进行采集监控
    controller.executionStats = workflowstore.NewExecutionStatsMonitor(scope.NewSubScope("execstats"), flyteworkflowInformer.Lister(), activeExecutions)

    // 创建上层WF执行器
    workflowExecutor, err := workflow.NewExecutor(ctx, store, controller.enqueueWorkflowForNodeUpdates, eventSink, controller.recorder, cfg.MetadataPrefix, nodeExecutor, &cfg.EventConfig, cfg.ClusterID, scope, activeExecutions)
    if err != nil {
        return nil, err
    }

    // 执行入口
    handler := NewPropellerHandler(ctx, cfg, store, controller.workflowStore, workflowExecutor, scope)
    // 将执行器管理至worker池
    controller.workerPool = NewWorkerPool(ctx, scope, workQ, handler)

    if cfg.EnableGrpcLatencyMetrics {
        grpc_prometheus.EnableClientHandlingTimeHistogram()
    }

    logger.Info(ctx, "Setting up event handlers")
    // Set up an event handler for when FlyteWorkflow resources change
    _, err = flyteworkflowInformer.Informer().AddEventHandler(controller.getWorkflowUpdatesHandler())
    if err != nil {
        return nil, fmt.Errorf("failed to register event handler for FlyteWorkflow resource changes: %w", err)
    }

    // 内存中保存pod模板
    updateHandler := flytek8s.GetPodTemplateUpdatesHandler(&flytek8s.DefaultPodTemplateStore)
    _, err = podTemplateInformer.Informer().AddEventHandler(updateHandler)
    if err != nil {
        return nil, fmt.Errorf("failed to register event handler for PodTemplate resource changes: %w", err)
    }

    return controller, nil
}
```

New创建后，后面的几行代码是正式启动服务器，其中Run函数，会启动controler、以及处理线程池、WF GC检查，和监控等等组件。

```go
func (c *Controller) Run(ctx context.Context) error {
    if c.leaderElector == nil {
        logger.Infof(ctx, "Running without leader election.")
        // 这里如果是多propeller实例的话，需要选举leader，如果为普通propeller，直接启动
        return c.run(ctx)
    }

    // 否则设置自己为leader
    logger.Infof(ctx, "Attempting to acquire leader lease and act as leader.")
    go c.leaderElector.Run(ctx)
    <-ctx.Done()
    return nil
}

func (le *LeaderElector) Run(ctx context.Context) {
    defer runtime.HandleCrash()
    defer le.config.Callbacks.OnStoppedLeading()

    // 此处获取锁，并向Prometheus上报host name
    if !le.acquire(ctx) {
        return // ctx signalled done
    }
    ctx, cancel := context.WithCancel(ctx)
    defer cancel()
    // 获取锁回调，进行业务信息更新, 调用run函数
    go le.config.Callbacks.OnStartedLeading(ctx)
    // 尝试向k8s获取锁，失败后上报Prometheus leader退出
    le.renew(ctx)
}

func (c *Controller) run(ctx context.Context) error {
    // Initializing WorkerPool
    logger.Info(ctx, "Initializing controller")
    if err := c.workerPool.Initialize(ctx); err != nil {
        return err
    }

    // Start the WF GC
    if err := c.gc.StartGC(ctx); err != nil {
        logger.Errorf(ctx, "failed to start background GC")
        return err
    }

    // Start the collector process
    c.levelMonitor.RunCollector(ctx)
    c.executionStats.RunStatsMonitor(ctx)

    // Start the informer factories to begin populating the informer caches
    logger.Info(ctx, "Starting FlyteWorkflow controller")
    return c.workerPool.Run(ctx, c.numWorkers, c.flyteworkflowSynced)
}

```

至此controller开始正式监听k8s事件和WF CRD，CRD会通过getWorkflowUpdatesHandler返回的informer回调触发controller的action

![flyte_propeller](/images/flyte/4/propeller-controller-new.png)

## 2. CRD驱动逻辑

