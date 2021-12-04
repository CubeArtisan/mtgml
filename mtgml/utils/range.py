class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __str__(self):
        return f'values in the inclusive range [{self.start}, {self.end}]'

    def __repr__(self):
        return str(self)
