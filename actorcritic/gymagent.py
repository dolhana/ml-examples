"""Generic gym agent"""

class Generic():
    """OpenAI gym agent"""

    def __init__(self, env, model_builder):
        """Creates an agent that interacts with the given env"""
        self.env = env
        self.last_state = None
        self.done = True
        self.model = model_builder(env)

    def step(self, auto_restart_episode=True):
        """agent.step() runs one step in the environment

        Args:
	        auto_restart_episode (bool):
        				If True, resets the env and runs a step if self.done == True.
        				If False, tries to run a step even if self.done == True.

        Returns:
        	(prev_state, action, reward, cur_state, done, info)
        """
        if self.done and auto_restart_episode:
            self.restart_episode()

        action = self.choose_action(self.last_state)
        state, reward, done, _info = self.env.step(action)

        return self.last_state, action, reward, state, done, _info

    def restart_episode(self):
        """Resets the environment to start a new episode

        Returns:
        	state:	a new state
        """
        self.last_state = self.env.reset()
        self.done = False

    def choose_action(self, state):
        """Choose an action based on the state

        Args:
        	state:	agent chooses an action based on this state

        Returns:
        	action
        """
        action = self.model.choose_action(state)
        return action
