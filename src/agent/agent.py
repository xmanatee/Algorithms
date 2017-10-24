import numpy as np

import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer

from utils.multi_probabilistic import MultiProbabilisticResolver
from utils.action_encoder import ActionEncoder

from agentnet.agent import Agent


def build_agent(action_shape, state_shape):
    observation_layer = InputLayer((None, *state_shape))

    net = DenseLayer(observation_layer, 10,
                     nonlinearity=lasagne.nonlinearities.sigmoid,
                     name='dense1')
    # net = DenseLayer(net, 256, name='dense2')

    # a layer that predicts Qvalues

    policy_layer_flattened = DenseLayer(
        net, num_units=np.prod(action_shape),
        nonlinearity=lasagne.nonlinearities.softmax,
        name="q-evaluator layer")

    policy_layer = ReshapeLayer(
        policy_layer_flattened,
        ([0], *action_shape)
    )

    V_layer = DenseLayer(
        net, 1, nonlinearity=None, name="state values")

    # Pick actions at random proportionally to te probabilities
    action_layer = MultiProbabilisticResolver(
        policy_layer,
        name="e-greedy action picker",
        assume_normalized=True)

    # print("ActionL: ", action_layer.output_shape)

    # action_layer = ActionEncoder(
    #     action_layer,
    #     base=3)

    # print("ActionL': ", action_layer.output_shape)

    # action_layer = T.printing.Print("A")(action_layer)

    # all together
    agent = Agent(observation_layers=observation_layer,
                  policy_estimators=(policy_layer_flattened, V_layer),
                  action_layers=action_layer)

    return agent, action_layer, V_layer
