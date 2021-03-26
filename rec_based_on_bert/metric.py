import torch

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

    def __init__(self, topK, strict=False):
        self.topK = topK
        self.reset()
        self.strict = strict

    def __call__(self, logits, target):
        if self.strict:
            _, pred = logits.topk(self.topK, 1, True, True)
            _, tar = target.topk(self.topK, 1, True, True)
            pred = pred.t()
            tar = tar.t()
            correct = pred.eq(tar)
            self.correct_k += correct[:self.topK].view(-1).float().sum(0)
        else:
            logits = torch.sigmoid(logits)
            logits[logits >= 0.5] = 1
            temp = logits.eq(target)
            for row in temp:
                self.correct_k += (True in row)


        self.total += target.size(0)

    # def __call__(self, logits, target):
    #     logits = torch.sigmoid(logits)
    #     logits[logits >= 0.5] = 1
    #     self.correct_k += logits.eq(target).sum().item()
    #     self.total += target.shape[0]

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        if self.strict:
            return self.correct_k / self.total / self.topK
        return self.correct_k / self.total

    def name(self):
        return 'acc'
