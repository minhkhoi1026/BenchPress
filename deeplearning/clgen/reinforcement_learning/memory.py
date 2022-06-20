class Memory(object):
  """
  Replay buffer of previous states and actions.
  """
  def __init__(self):
    self.action_buffer = []
    self.state_buffer  = []
    self.reward_buffer = []
    self.loadCheckpoint()
    return

  def add(self, action, state, reward) -> None:
    """Add single step to memory buffers."""
    self.action_buffer.append(action)
    self.state_buffer.append(state)
    self.reward_buffer.append(reward)
    return
  
  def loadCheckpoint(self) -> None:
    """Fetch memory's latest state."""
    return
  
  def saveCheckpoint(self) -> None:
    """Save Checkpoint state."""
    return