class metric_accuracy():
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    Examples:
        >>> metric = Accuracy(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self, topK):
        self.topK = topK
        self.reset()

    def __call__(self, logits, target):
        _, pred = logits.topk(self.topK, 1, True, True)
        _, tar = target.topk(self.topK, 1, True, True)
        pred = pred.t()
        tar = tar.t()
        correct = pred.eq(tar)
        self.correct_k += correct[:self.topK].view(-1).float().sum(0)
        self.total += target.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k) / self.total

    def name(self):
        return 'acc'
