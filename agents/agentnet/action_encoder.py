import theano.tensor as T

from agentnet.resolver import BaseResolver


class ActionEncoder(BaseResolver):

    def __init__(self, incoming, base, output_dtype='int32',
                 name='ActionEncoder'):
        self.base = base

        super(ActionEncoder, self).__init__(
            incoming,
            name=name,
            output_dtype=output_dtype)

    def get_output_for(self, policy, greedy=False, **kwargs):
        # policy = T.printing.Print("policy: ")(policy)
        actions_num = policy.shape[-1]
        base_pow = T.pow(
            self.base,
            T.arange(0, actions_num, dtype=self.output_dtype))

        # base_pow = T.printing.Print("base_pow: ")(base_pow)

        # base_pow = T.printing.Print(base_pow)
        output = T.dot(policy, base_pow)
        # output = T.printing.Print("output: ")(output)
        return output

    def get_output_shape_for(self, input_shape):
        """returns shape of layer output"""
        output_shape = input_shape[:-1]
        return output_shape
