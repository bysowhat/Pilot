import abc

class Dataset(metaclass=abc.ABCMeta):
  def __init__(self):
    pass

  @abc.abstractmethod
  def __iter__(self):
    pass

  @abc.abstractmethod
  def __len__(self):
    pass