---
layout: post
title: NodeJS多线程库Threads
date: 2021-02-05 11:56:24.000000000 +08:00
---

## 前言

threads.js 是个风格良好的JS worker库，它不仅对NodeJS原生的worker线程做了优雅的封装，并且可在版本NodeJS 8 - 12、Web browser中使用，同时提供给用户统一的体验。

作者[Andy Wermke](https://andywer.com/)拥有丰富的NodeJS开发经验，因此threads.js有自己独特的风格和编码规范，很值得学习和借鉴。

threads.js主要有一下几个功能:

* 多种worker加载模式: 路径加载、Blob加载
* 支持Typescript、Webpack、Electron、Parcel下使用threads.js
* 支持Observer 和 Promise 两种异步调用模式
* 支持线程池调用，操作简单，并有丰富的参数配置
* 更简单的Transferable数据封装，支持对类进行序列化操作
* 支持子线程和主线程间的事件订阅、发布

我将从threads的整体架构出发，详细介绍threads的设计思想和实现细节，我们这就开始。

## Threads.js 整体设计

![Threads线程设计](/images/threads/native_thread_wrap.png)

threads通过两个代理函数spawn 和 expose, 对master线程和worker线程做了简洁的封装.

其中:

* spawn是主线程调用，用来对woker进行一些设置，包括初始化、event事件回调、执行函数参数封装、注册销毁函数等操作，并已Object的形式进行返回，方便主线程的调用。

* expose是工作线程调用，用来对执行函数或者模块进行封装。包括订阅master线程信息，执行worker函数。

对于worker的返回结果，Threads不仅支持Promise结果，也支持自己实现的Observer，主线程通过订阅Observer可以连续获取worker线程的中间值和返回值，并且可以实现多端订阅，方便主线程各个模块同步工作线程信息。

另外，threads支持对类进行序列化和反序列化，这在复杂消息传递时，尤为方便。

本文，会从Thread Worker设计入手，逐步展开，从代码级别了解主线程、工作线程消息同步机制，线程池如何良好的工作，平台透明化，序列化与反序列化。

## 线程封装

Threads在master端和worker端对原始的native thread做了封装， 使其使用更简单，自动初始化检查，搜集运行错误，并扩展出新的feature等

### 工作线程封装函数: expose

expose函数，是对执行函数或者模块的封装，我们先简要看一下在worker线程的JS中，expose的使用方法。

```typescript
import { expose } from "threads/worker"

let currentCount = 0

// Function
export const func_example = async (input_praram: string) => { 
  // do something 
} 

// Object
const counter = {
  getCount() {
    return currentCount
  },
  increment() {
    return ++currentCount
  },
  decrement() {
    return --currentCount
  }
}

export type Counter = typeof counter

expose(counter) // or expose(func_example)
```
 可以看出，`expose`实际上的输入可以试一个Object或者一个function, 如果定义的是Object，那么需要实现所属方法，供master通过引用调用; 如果是function，则直接调用即可。

 在expose函数内部，通过`subscribeToMasterMessages`方法来订阅主线程的发过来的信息，实现原理是通过在工作线程的parentPort上注册`message`事件来实现现，`parentPort`是master线程和worker线程通信的底层管道接口，可实现双工通信。如果worker线程需要发送信息，则通过`postMessage`发送信息到主线程。

 在接收到主线程发过来的运行信息后，会触发`runFunction`函数，异步执行worker函数，并获取函数执行的Promise或者Observer。

```typescript
// if is is a Observer result, then subscribe the master info for asynchronous communication.
if (isObservable(syncResult)) {
    const subscription = syncResult.subscribe(
      value => postJobResultMessage(jobUID, false, serialize(value)),
      error => {
        postJobErrorMessage(jobUID, serialize(error) as any)
        activeSubscriptions.delete(jobUID)
      },
      () => {
        postJobResultMessage(jobUID, true)
        activeSubscriptions.delete(jobUID)
      }
    )
    activeSubscriptions.set(jobUID, subscription)
  } else {
    // if it is a Promise result, wait the 
    try {
      const result = await syncResult
      postJobResultMessage(jobUID, true, serialize(result))
    } catch (error) {
      postJobErrorMessage(jobUID, serialize(error) as any)
    }
  }
```

这里值得注意点一点是，相比于Promise的一次性的`postMessageToMaster`结果到主线程，Observer因为拥有自身的队列，可以同时对多个subscriber发布信息，并且可以不定时的`postMessageToMaster`，例如可以应用于数据流等，具体可参考代码。

### 主线程封装函数: spawn

spawn函数，是在主线程中对worker模块进行封装，在本例中，先简要看一下调用方法。

```typescript
import { spawn, Thread, Worker } from "threads"
import { Counter } from "./workers/counter"

const counter = await spawn<Counter>(new Worker("./workers/counter"))
console.log(`Initial counter: ${await counter.getCount()}`)

await counter.increment()
console.log(`Updated counter: ${await counter.getCount()}`)

// or
let result = await counter("test string")
console.log(`Example function: ${result}`)

await Thread.terminate(counter)
```

spawn负责检查worker初始化结果，订阅worker发布的事件信息，触发worker运行，创建worker终止方法等。

```typescript
const initMessage = await withTimeout(receiveInitMessage(worker), timeout, `Timeout: Did not receive an init message from worker after ${timeout}ms. Make sure the worker calls expose().`)
  const exposed = initMessage.exposed

  const { termination, terminate } = createTerminator(worker)
  const events = createEventObservable(worker, termination)
  // find the different type for expose worker thread
  if (exposed.type === "function") {
    const proxy = createProxyFunction(worker)
    return setPrivateThreadProps(proxy, worker, events, terminate) as ExposedToThreadType<Exposed>
  } else if (exposed.type === "module") {
    const proxy = createProxyModule(worker, exposed.methods)
    return setPrivateThreadProps(proxy, worker, events, terminate) as ExposedToThreadType<Exposed>
  } else {
    const type = (exposed as WorkerInitMessage["exposed"]).type
    throw Error(`Worker init message states unexpected type of expose(): ${type}`)
  }
```

首先创建worker销毁函数,  其中`termination`负责对worker销毁方法，`terminate`是一个Promise对象负责监控worker销毁，并通知主线程销毁结果。

然后创建工作线程事件的Observer用于获取工作线程的`message`和`error` 。

接下来就是创建代理函数对工作了，代理函数锁做就是返回一个闭包函数，在函数里会对执行参数进行封装，并`postMessage`给工作线程，并订阅worker线程消息，返回消息的Observer给主线程的调用方。`createProxyModule`会把Object都方法进行拆解，最后依然调用`createProxyFunction`进行同样的操作。

最后通过`setPrivateThreadProps`组装所有的内容，返回给主线程进行调用, 这里包括error event、result event、terminate function、exposed worker。

```typescript
return Object.assign(raw, {
    [$errors]: workerErrors,
    [$events]: workerEvents,
    [$terminate]: terminate,
    [$worker]: worker
  })

```

## 线程间信息传递

工作线程可以通过NodeJS的底层管道机制进行通信，Threads对传统的通信方式进行了扩展，从而支持了类的序列化，共享数据的封装，并可以使用Observer和Promise友好、灵活的在master和worker之间进行通信。

### 序列化

主线程和工作线程是通过[Structured clone algorithm](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm)来进行复制传递的，当然为了提升传输效率，对于Binary的数据通过封装在`Transferable`结构中，使用共享内存的方式共享数据，提升性能，支持的数据结构有`ArrayBuffer`, `MessagePort`, `ImageBitmap`, `OffscreenCanvas`。

有时对于传递复杂的结构体或类时，**Structured clone algorithm**就无能为力了，因为对类进行序列化和反序列化也是十分必要的。

Threads通过对实现序列化接口，来自定义序列化和反序列化，最后通过简单的注册代码，即可引入自定义序列化，使用还是比较方便的。

### 信息通知

原生的信息通知，是通过注册不同的事件来相互通信的，对于数据的分发、处理，多端订阅等等，并没有涉及。所以threads通过Observer的方式支持多个订阅端，对同一个信息的监控，因此通过订阅的方式解偶了复杂的逻辑中的消息传递。

作者实现了Subject类，他是个单例类，可以同时注册多个的Observer，Subject可以向每一个Observer的Subscriber广播数据，从而实现了多端订阅，代码解偶。

```typescript
export class Subject<T> extends Observable<T> implements ObservableLike<T> {
  private [$observers]: Array<SubscriptionObserver<T>>

  constructor() {
    super(observer => {
      this[$observers] = [
        ...(this[$observers] || []),
        observer
      ]
      const unsubscribe = () => {
        this[$observers] = this[$observers].filter(someObserver => someObserver !== observer)
      }
      return unsubscribe
    })

    this[$observers] = []
  }
  // close the subscriber channel
  public complete() {
    this[$observers].forEach(observer => observer.complete())
  }

  public error(error: any) {
    this[$observers].forEach(observer => observer.error(error))
  }
  // publish the value to subscriber
  public next(value: T) {
    this[$observers].forEach(observer => observer.next(value))
  }
}
```

从代码中，可以清楚的看出Subject其实就是多个Observers的集合，每个observer都可以根据代码单独的被subscribe。Observer的实现细节我将在下一个文章中剖析。

## 线程池

在实际使用中，常用的就是线程池的使用了。有了以上的基础，我们来看一下线程池是如何工作的。先挂上线程池的完整架构图:

![](/images/threads/thread-pool-structure.png)

我们可以看到线程池新增了几个组成部分: Pool, , Event Observer, Task,  Queue,  Scheduler。结合原有的worker，最终组成了线程池。

### Task

线程池本质上是对Task集合分配worker线程进行处理，这里的Task的核心是一个回调函数，运行在主线程，负责对worker线程的输入输出进行处理。

Task实现了`QueuedTask`接口，其中包括，id、run函数、cancel函数、then函数。

其中:

* id为task id，为从1开始递增的int整形。
* run函数，即处理worker输入输出处理函数的回调函数。回调参数为目前处理次Task的worker线程。
* cancel函数，该函数可对task自身做撤销操作，把task id从队列中移除，并通知主线程。
* then函数，该函数返回一个Promise，当该Task处理完毕，或者出现异常时，会回调该Promise，从而通知主线程。

### Pool

Pool在构造函数中会根据线程池参数中设定的线程个数，创建和初始化每个worker线程，关于worker线程如何进行初始化，请参照上节的spawn部分。同时也会构造event observer，提供给调用者接收事件信息，比如Task运行状态、线程池的初始化、失败错误等。

Pool的另外一个方法是`queue(taskFunc: (worker) => void): QueueTask`，它的功能是构造一个task函数作为入队线程池，并返回`QueueTask`实例，供调用方使用。

每次都入队操作，最后都会触发`Scheduler`的运行，从队列中取出Task进行调度。

### Event Observer

Event Observer运行在主线程，主要任务是接收线程池内部的信息和Task状态信息，并以统一的出口通知主线程，Observer的具体运行原理，我讲在下一篇文章中详细描述。

### Queue

线程池必备组件，它的实现比较简单，是一个数组，用于存储主线程的task。

### Scheduler

Scheduler的大致逻辑是，从Queue中取出一个Task，并从Pool中找到目前可用的worker线程，并回调task上的run函数，将worker线程暴露给调用者，调用者来决定如何使用worker，run方法最后把worker都结果返回给event observer，从而通知主线程。次过程一直循环，指导队列中无task处理，等待下一次queue后，重新触发scheduler运行。

## 总结

本文从ThreadsJS整体出发，介绍了Thread封装、通信和线程池的设计与实现。该库代码可读性高，设计灵活。另外作者typescript使用方面独具一格，尤其对接口、方法的type定义，很有借鉴意义，是个很好的学习Typescript和NodeJS的库，非常适合入门者进阶学习。

在实际生产环境中，已证明该库比较稳定可靠，欢迎小伙伴们在JS复杂的多线程场景中使用，会有事半功倍的效果。
