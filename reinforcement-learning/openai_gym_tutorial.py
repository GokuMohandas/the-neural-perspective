import gym
import tensorflow as tf
import numpy as np

class parameters(object):

    def __init__(self):

        self.num_episodes = 1000
        self.dim_observation = 4
        self.dim_actions = 2

        self.num_hidden_units = 2
        self.learning_rate = 0.1
        self.discount_factor = 0.995

def random_play(num_episodes):
    """
    Simulation of the cartpole game with
    random movements.
    """

    for i_episode in range(num_episodes):

        # Reset the environment
        print "\nEpisode %i" % i_episode
        observation = env.reset()

        total_reward = 0
        num_time_steps = 0
        while True:

            # Render the new environment
            env.render()
            print(observation)

            # random action sampled from action space
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            total_reward += reward
            num_time_steps += 1

            # done = True if the pole falls or time is up
            if done:
                print("Episode finished after %i timesteps "
                      "with a total reward of %i." %
                      (num_time_steps, total_reward))
                break

def discount_rewards(rewards, gamma):
    """
    Return discounted rewards weighed by gamma.
    Each reward will be replaced with a weight reward that
    involves itself and all the other rewards occuring after it.
    The later the reward after it happens, the less effect it
    has on the current rewards's discounted reward since gamma<1.

    [r0, r1, r2, ..., r_N] will look someting like:
    [(r0 + r1*gamma^1 + ... r_N*gamma^N), (r1 + r2*gamma^1 + ...), ...]
    """
    return np.array([sum([gamma**t*r for t, r in enumerate(rewards[i:])])
        for i in range(len(rewards))])

class agent(object):

    def __init__(self, FLAGS):

        # Placeholders
        self.observations = tf.placeholder(name="observations",
            shape=[None, FLAGS.dim_observation], dtype=tf.float32)
        self.actions = tf.placeholder(name="actions",
            shape=[None, 1], dtype=tf.float32)
        self.rewards = tf.placeholder(name="rewards",
            shape=[None, 1], dtype=tf.float32)

        # Net
        with tf.variable_scope('net'):
            W1 = tf.get_variable(name='W1',
                shape=[FLAGS.dim_observation, FLAGS.num_hidden_units])
            W2 = tf.get_variable(name='W2',
                shape=[FLAGS.num_hidden_units, 1])
            z1 = tf.matmul(self.observations, W1)
            fc1 = tf.nn.relu(z1)
            z2 = tf.matmul(fc1, W2)
            self.fc2 = tf.nn.sigmoid(z2)

        # Loss
        self.loss = - tf.reduce_mean((self.actions*tf.log(self.fc2) +
            (1-self.actions)*(tf.log(1 - self.fc2)))*self.rewards, 0)

        # Optimizing
        self.train_optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        self.train_step = self.train_optimizer.minimize(self.loss)



if __name__ == '__main__':

    # Set the gym environment
    env = gym.make('CartPole-v0')
    accum_rewards = np.zeros(100)

    # random moves
    random_play(num_episodes=1)

    # RL model parameters
    FLAGS = parameters()

    with tf.Session() as sess:
        model = agent(FLAGS)
        sess.run(tf.initialize_all_variables())

        for episode in range(FLAGS.num_episodes):

            # reset the environment for each episode
            print "\nEpisode %i" % episode
            observation = env.reset()

            # store variables
            actions, rewards, states = [], [], []

            while True:

                # show visual
                env.render()

                states.append(observation)
                fc2 = sess.run(model.fc2, feed_dict={
                    model.observations: np.reshape(observation,
                        [1, FLAGS.dim_observation])})

                # Determine the action
                probability = fc2[0][0]
                action = int(np.random.choice(2, 1,
                    p = [1-probability, probability]))

                observation, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)

                # Episode is finished (task failed)
                if done:
                    epr = np.vstack(
                        discount_rewards(rewards, FLAGS.discount_factor))
                    eps = np.vstack(states)
                    epl = np.vstack(actions)

                    epr -= np.mean(epr)
                    epr /= np.std(epr)
                    sess.run(model.train_step,
                        feed_dict = {model.observations: eps,
                        model.actions: epl, model.rewards: epr})

                    accum_rewards[:-1] = accum_rewards[1:]
                    accum_rewards[-1] = np.sum(rewards)

                    # average reward for last 100 steps
                    print('Running average steps:',
                        np.mean(accum_rewards[accum_rewards > 0]),
                        'Episode:', episode+1)

                    break











