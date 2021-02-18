---
layout: post
title: NodeJS观察者模式库(observable-fns)
date: 2021-02-18 13:07:24.000000000 +08:00
---

> 平时在开发过程中，经常会用到设计模式(Design pattern), 它是前人总结出的宝贵经验和经典实践。常用的设计模式有三类：创建型模式（工厂模式、单例模式、建造者模式）、结构型模式（装饰器模式、适配器模式、代理模式）、行为型模式（监听者模式、观察者模式)。本文介绍一种观察者模式的实现库:Observable-fns, 使用者可以快速在NodeJS环境下使用该模式进行消息分发和订阅。

<!-- TOC -->

- [前言](#%E5%89%8D%E8%A8%80)
- [观察模式整体设计](#%E8%A7%82%E5%AF%9F%E6%A8%A1%E5%BC%8F%E6%95%B4%E4%BD%93%E8%AE%BE%E8%AE%A1)
- [广播消息](#%E5%B9%BF%E6%92%AD%E6%B6%88%E6%81%AF)
- [其他消息管理](#%E5%85%B6%E4%BB%96%E6%B6%88%E6%81%AF%E7%AE%A1%E7%90%86)
- [总结](#%E6%80%BB%E7%BB%93)

<!-- /TOC -->

## 前言

`Observable-fns`是一个实现比较巧妙的Observer模式库。该库的作者同样为[Andy Wermke](https://andywer.com/), `Observable-fns`不仅提供了传统的发布订阅接口，而且还为使用者提供了常用的工具函数，方便开发者进行消息进行过滤、扫描、变换、导入导出、合并订阅等，是非常使用的工具库。

## 观察模式整体设计

在观察者模式中，数据通过观察者作为输入端，想订阅者传递。`Observable-fns`实现了`Observable`类，其中包括subscriber的订阅回调方法，数据管理方法等。具体架构可参看下图:

![Observable-fns设计](/images/observer/observer.jpg)

从图中的**Step 1**中，我们可以看出，创建`Observable`类时，需要传入**订阅回调函数**，这个函数是订阅者调用`subscribe`方法时触发调用的。

当订阅者调用`subscribe`时，就会创建一个`Subscription`类，它拥有自己的消息队列、订阅状态、以及会生成一个订阅管理类`SubscriptionObserver`，它是对订阅者`Subscription`类的管理类，作用是操作`Subscription`类进行消息分发，并通过内部状态机逻辑，维护`Subscription`的状态，这是**Step 2**所做的事情.

接着，`Subscription`类会执行**Step 3**调用**订阅回调函数**，参数就是上一步创建的`SubscriptionObserver`类， 回到函数内部来定义用于自定义的消息对接和发布代码。

## 广播消息

在整体设计中，我们实现了观察者和订阅者的一一映射，并建立了信息的发布、订阅通道，并可以管理消息发布过程中的订阅状态。

但是如何实现单一观察者发布消息，多方订阅者接收消息的场景呢？`Observable-fns`也给出了自己的方案:**multicast**.

![Observable multicast](/images/observer/multicast_subscriber.jpg)

`multicast`是一个函数，输入参数是我们上节中创建的`Observable`类, 函数内部会创建一个`MulticastSubject`类，并返回一个`ObservableProxy`类， `ObservableProxy`会对`MulticastSubject`进行管理，我们可以参考代码实现。

```typescript
function multicast<T>(coldObservable: ObservableLike<T>): Observable<T> {
  const subject = new Subject<T>()

  let sourceSubscription: ReturnType<ObservableLike<T>["subscribe"]> | undefined
  let subscriberCount = 0

  return new Observable<T>(observer => {
    // Init source subscription lazily
    if (!sourceSubscription) {
      sourceSubscription = coldObservable.subscribe(subject)
    }

    // Pipe all events from `subject` into this observable
    const subscription = subject.subscribe(observer)
    subscriberCount++

    return () => {
      subscriberCount--
      subscription.unsubscribe()

      // Close source subscription once last subscriber has unsubscribed
      if (subscriberCount === 0) {
        unsubscribe(sourceSubscription)
        sourceSubscription = undefined
      }
    }
  })
}

```

这里的`Observable`类我们定义为**cold observer**， 因为这时只是定义好订阅回调方法，单并没有触发。通过`multicast`方法的调用，会触发一次`Observable`的subscribe调用，从而初始化订阅通道，此时`Observable`类就变成了**hot observer**。

当订阅者使用`ObservableProxy`的subscribe方法时，会触发`MulticastSubject`的subscribe方法，`MulticastSubject`继承了`Observable`类，并维护了一个**Observer set**， 每当调用一次subscribe方法时，即可生成一个`SubscriptionObserver`并缓存起来。

当`Observable`通过next方法获取到消息输入时，会传递给`MulticastSubject`的next方法，最终`MulticastSubject`将该消息分发给**Observer set**中的所有**Observer**， 没有**Observer**都拥有自己的消息队列以及**subscriber**. 因此最终行程了一个原始**Observer**，多个**subscriber**的分发模式。

## 其他消息管理

除了常用的**multicast**外，`Observable-fns`还提供了merge、scan、map、filter、flatMap、interval等方法。基本思路都是新建一个`ObservableProxy`，在这个新的`Observer`中进行消息的过滤、组合、转换、分发。具体的代码细节可以参考[源码](https://github.com/andywer/observable-fns/tree/master/src)。

这里方法中大多用到了一个异步调度器(`AsyncSerialScheduler`), 它的功能可以异步接收消息，并串行执行消息处理，从而提高性能，实现代码如下:

```typescript
class AsyncSerialScheduler<T> {
  private _baseObserver: SubscriptionObserver<T>
  private _pendingPromises: Set<Promise<any>>

  constructor(observer: SubscriptionObserver<T>) {
    this._baseObserver = observer
    this._pendingPromises = new Set()
  }

  complete() {
    Promise.all(this._pendingPromises)
      .then(() => this._baseObserver.complete())
      .catch(error => this._baseObserver.error(error))
  }

  error(error: any) {
    this._baseObserver.error(error)
  }

  schedule(task: (next: (value: T) => void) => Promise<void>) {
    const prevPromisesCompletion = Promise.all(this._pendingPromises)
    const values: T[] = []

    const next = (value: T) => values.push(value)

    const promise = Promise.resolve()
      .then(async () => {
        await prevPromisesCompletion
        await task(next)
        this._pendingPromises.delete(promise)

        for (const value of values) {
          this._baseObserver.next(value)
        }
      })
      .catch(error => {
        this._pendingPromises.delete(promise)
        this._baseObserver.error(error)
      })

    this._pendingPromises.add(promise)
  }
}
```

从代码中我们可以看出， `AsyncSerialScheduler`拥有三个方法`schedule`、`error`、`complete`.

* `schedule`: 该方法接收task输入，task是一个函数，用于具体功能逻辑的实现，如果达到了该功能的目的，则此消息放入`values`队列中，然后发送给目标`Subjecter`。上述操作是异步执行，所以当有I/O操作时，不会影响性能。
* `error`: 是向目标`Subjecter`传递error信息。
* `complete`: 是一个同步方法，它的作用是等待直到所有`task`都被执行完毕。方便使用者可以获得任务结束点的通知。

## 总结

以上就是`Observable-fns`的核心内容，主要介绍了`Observer` -> `Subscriber`的核心代码实现，并分析了在**1vN**场景下，**multicast**的实现方法。`Observable-fns`的设计和代码对NodeJS开发者的技能提高帮助很大，有很多值得学习的地方，可以在平时的学习和工作中借鉴使用。希望本文可以对广大NodeJS读者的提升有所帮助。
