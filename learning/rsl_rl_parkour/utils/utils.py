from collections import OrderedDict
import numpy as np
import torch

def get_obs_slice(segments: OrderedDict, component_name: str):
    """ Get the slice from segments and name. Return the slice and component shape """
    obs_start = obs_end = 0
    component_shape = None
    for k, v in segments.items():
        obs_start = obs_end
        obs_end = obs_start + np.prod(v)
        if k == component_name:
            component_shape = v # tuple
            break
    assert component_shape is not None, "No component ({}) is found in the given components {}".format(component_name, [segments.keys()])
    return slice(obs_start, obs_end), component_shape

def get_subobs_size(obs_segments, component_names):
    """ Compute the size of a subset of observations. """
    obs_size = 0
    for component in obs_segments.keys():
        if component in component_names:
            obs_slice, _ = get_obs_slice(obs_segments, component)
            obs_size += obs_slice.stop - obs_slice.start
    return obs_size

def get_subobs_by_components(observations, component_names, obs_segments):
    """ Get a subset of observations from the full observation tensor. """
    estimator_input = []
    for component in obs_segments.keys():
        if component in component_names:
            obs_slice, _ = get_obs_slice(obs_segments, component)
            estimator_input.append(observations[..., obs_slice])
    return torch.cat(estimator_input, dim= -1) # NOTE: this is a 2-d tensor with (batch_size, obs_size)

def substitute_estimated_state(observations, target_components, estimated_state, obs_segments):
    """ Substitute the estimated state into part of the observations.
    """
    estimated_state_start = 0
    for component in obs_segments:
        if component in target_components:
            obs_slice, obs_shape = get_obs_slice(obs_segments, component)
            estimated_state_end = estimated_state_start + np.prod(obs_shape)
            observations[..., obs_slice] = estimated_state[..., estimated_state_start:estimated_state_end]
            estimated_state_start = estimated_state_end
    return observations