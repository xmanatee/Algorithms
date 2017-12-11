# get reference Qvalues according to Qlearning algorithm
import theano
import lasagne
from agentnet.learning import a2c
from theano import tensor as T


def get_train_step(policy_seq, V_seq, weights, replay):
    # Train via actor-critic (see here - https://www.youtube.com/watch?v=KHZVXao4qXs)

    policy_seq = T.printing.Print(">>>> policy_seq; ")(policy_seq)
    V_seq = T.printing.Print(">>>> V_seq; ")(V_seq)
    ### COMMENTED ASSERT ACTION.DIM==2 LINE IN THIS METHOD
    elwise_mse_loss = a2c.get_elementwise_objective(
        policy_seq,
        V_seq[:, :, 0],
        replay.actions[0],
        replay.rewards,
        replay.is_alive,
        gamma_or_gammas=0.99)

    # compute mean over "alive" fragments
    loss = elwise_mse_loss.sum() / replay.is_alive.sum()

    reg = T.mean((1. / policy_seq).sum(axis=-1))
    loss += 0.01 * reg

    # Compute weight updates
    updates = lasagne.updates.rmsprop(loss, weights, learning_rate=0.001)
    # updates = lasagne.updates.adam(loss, weights, learning_rate=0.001)

    # compile train function

    train_step = theano.function([], loss, updates=updates)

    return train_step
