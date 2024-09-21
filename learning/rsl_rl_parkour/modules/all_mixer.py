from rsl_rl.modules import ActorCriticRecurrent
from .state_estimator import EstimatorMixin
from .encoder_actor_critic import EncoderActorCriticMixin

class EncoderStateAcRecurrent(EstimatorMixin, EncoderActorCriticMixin, ActorCriticRecurrent):
    pass

