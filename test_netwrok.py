import theano
import theano.tensor as T 
import numpy
import numpy as np 
import lasagne
from collections import OrderedDict

l_in = lasagne.layers.InputLayer(
        shape=(4,84,84)
    )

l_in = lasagne.layers.ReshapeLayer(l_in, (1, 4,84,84))

l_in_out = l_in.get_output_shape_for((4,84,84))

#l_conv1 = dnn.Conv2DDNNLayer(
l_conv1 = lasagne.layers.Conv2DLayer(
    incoming=l_in,
    num_filters=16,
    filter_size=(8, 8),
    stride=(4, 4),
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.HeUniform(), # Defaults to Glorot
    b=lasagne.init.Constant(.1)
    #dimshuffle=True
)

l_conv1_out = l_conv1.get_output_shape_for(l_in_out)
print "L1:", l_conv1_out

l_conv2 = lasagne.layers.Conv2DLayer(
    incoming=l_conv1,
    num_filters=32,
    filter_size=(4, 4),
    stride=(2, 2),
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.HeUniform(),
    b=lasagne.init.Constant(.1)
    #dimshuffle=True
)

l_conv2_out = l_conv2.get_output_shape_for(l_conv1_out)
print "L1:", l_conv2_out

l_hidden1 = lasagne.layers.DenseLayer(
    incoming=l_conv2,
    num_units=256,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.HeUniform(),
    b=lasagne.init.Constant(.1)
)

l_hidden1_out = l_hidden1.get_output_shape_for(l_conv2_out)
print l_hidden1_out

#policy
l_policy = lasagne.layers.DenseLayer(
    incoming=l_hidden1,
    num_units=3,
    nonlinearity= lasagne.nonlinearities.softmax,
    #W=lasagne.init.HeUniform(),
    #b=lasagne.init.Constant(.1)
)

l_p_in = lasagne.layers.InputLayer((1,256))

l_alt_policy = lasagne.layers.DenseLayer(
    incoming=l_p_in,
    num_units=3,
    nonlinearity= lasagne.nonlinearities.softmax,
    #W=lasagne.init.HeUniform(),
    #b=lasagne.init.Constant(.1)
)




print l_policy.get_output_shape_for(l_hidden1_out)

l_value = lasagne.layers.DenseLayer(
    incoming=l_hidden1,
    num_units=1,
    nonlinearity=None,
    #W=lasagne.init.HeUniform(),
    #b=lasagne.init.Constant(.1)
)         


#  lasagne.layers.helper.get_all_params(l_hidden1) + l_value.get_params() == lasagne.layers.helper.get_all_params(l_value)



ix = T.tensor3('state')
a = T.iscalar('action')
p = T.fvector('prob')
r = T.fscalar('reward')


tmp_ix = numpy.random.rand(4,84,84).astype(numpy.float32)
lasagne.layers.get_output(l_policy, ix)

shared_out = lasagne.layers.get_output(l_hidden1, ix)
alt_policy_out = lasagne.layers.get_output(l_alt_policy, shared_out)
alt_policy = theano.function([ix], alt_policy_out)

policy_out = lasagne.layers.get_output(l_policy, ix)
value_out = lasagne.layers.get_output(l_value, ix)

policy = theano.function([ix], policy_out)
value = theano.function([ix], value_out)



log_policy = theano.function([ix, a],T.log(policy_out[0][a])) 


entropy = theano.function([ix], T.tensordot(policy_out,T.log(policy_out)))



# Loss function that we'll differentiate to get the policy gradient
# Note that we've divided by the total number of timesteps
p_loss = T.sum(T.log(policy_out))
p_params = lasagne.layers.helper.get_all_params(l_policy)

grads = T.grad(p_loss, p_params)
# Perform parameter updates.
# I find that sgd doesn't work well
# updates = sgd_updates(grads, params, stepsize)

get_grad = theano.function([ix],grads)


#self.pg_update = theano.function([ob_no, a_n, adv_n, stepsize], [], updates=updates, allow_input_downcast=True)
#self.compute_prob = theano.function([ob_no], prob_na, allow_input_downcast=True)


def rmsprop_updates(grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    updates = OrderedDict()
    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)
    c = 0
    for param, grad in zip(params, grads):
        print c 
        value = param.get_value(borrow=True)
        accu = theano.shared(numpy.zeros(value.shape, dtype=value.dtype),broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        mid_up = param - (learning_rate * grad / (T.sqrt(accu_new + epsilon)))
        try:
            updates[param] = lasagne.updates.norm_constraint( mid_up , 40 , 0)
        except:
            updates[param] = mid_up
        c+=1
    return updates


updates = rmsprop_updates(grads, p_params)


train_policy = theano.function(grads, [], updates=updates)


computed_grad = get_grad(tmp_ix)
train_policy(*computed_grad)

updates[param] = lasagne.updates.norm_constraint( mid_up , 40 ,  )



"""
------------------------------------------------------------------------------------------
"""
### mini batcch accumulation check 

b_in = lasagne.layers.InputLayer(
        shape=(5,4,84,84)
    )
#l_conv1 = dnn.Conv2DDNNLayer(
b_conv1 = lasagne.layers.Conv2DLayer(
    incoming=b_in,
    num_filters=16,
    filter_size=(8, 8),
    stride=(4, 4),
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.HeUniform(), # Defaults to Glorot
    b=lasagne.init.Constant(.1)
    #dimshuffle=True
)

b_conv2 = lasagne.layers.Conv2DLayer(
    incoming=b_conv1,
    num_filters=32,
    filter_size=(4, 4),
    stride=(2, 2),
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.HeUniform(),
    b=lasagne.init.Constant(.1)
    #dimshuffle=True
)

b_hidden1 = lasagne.layers.DenseLayer(
    incoming=b_conv2,
    num_units=256,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.HeUniform(),
    b=lasagne.init.Constant(.1)
)

print b_hidden1.get_output_shape_for((5,4,84,84))

#policy

b_p_in = lasagne.layers.InputLayer((5,256))

b_policy = lasagne.layers.DenseLayer(
    incoming=b_p_in,
    num_units=3,
    nonlinearity= lasagne.nonlinearities.softmax,
    #W=lasagne.init.HeUniform(),
    #b=lasagne.init.Constant(.1)
)

b_value = lasagne.layers.DenseLayer(
    incoming=b_p_in,
    num_units=1,
    nonlinearity= None,
    #W=lasagne.init.HeUniform(),
    #b=lasagne.init.Constant(.1)
)


bix = T.tensor4('state')
ba = T.ivector('action')
br = T.fvector('reward')


b_shared_out = lasagne.layers.get_output(b_hidden1, bix) 
bpolicy_out = lasagne.layers.get_output(b_policy, b_shared_out)
bvalue_out = lasagne.layers.get_output(b_value, b_shared_out)

bpolicy = theano.function([bix], bpolicy_out)

N = bix.shape[0]

# take log policy loss 
policy_loss = -T.log(bpolicy_out[T.arange(N), ba]) * br

# take entropy and add with the regularizer
entropy = - T.sum(- bpolicy_out * T.log(bpolicy_out), axis=1)

# add regullazrization
policy_loss += 0.01 * entropy

#policy_loss = T.sum(policy_loss)

# get the value loss
value_loss = (br - T.reshape(bvalue_out,(5,))**2)/2
#value_loss = T.sum(value_loss)


total_loss = T.sum(policy_loss + (0.5 * value_loss))


bp_params = lasagne.layers.helper.get_all_params(b_hidden1) + lasagne.layers.helper.get_all_params(b_policy)

bgrads = T.grad(total_loss, bp_params)
# Perform parameter updates.
# I find that sgd doesn't work well
# updates = sgd_updates(grads, params, stepsize)


b_tmp_ix = numpy.random.rand(5,4,84,84).astype(numpy.float32)

bupdates = rmsprop_updates(bgrads, bp_params)

b_train = theano.function([bix, ba, br],[], updates = bupdates , allow_input_downcast=True)
b_loss = theano.function([bix, ba, br],[total_loss],  allow_input_downcast=True)


entropy = - (bpolicy_out * T.log(bpolicy_out))
b_loss_1 = theano.function([bix, ba, br],[policy_loss],  allow_input_downcast=True)


b_loss_2 = theano.function([bix],[bvalue_out],  allow_input_downcast=True)
b_loss_2 = theano.function([bix, br, ba],[T.sum(policy_loss + (0.5 * value_loss))],  allow_input_downcast=True)

b_loss_2 = theano.function([bix, br, ba],total_loss,  allow_input_downcast=True)

b_loss_3 = theano.function([bix, br, ba],bgrads,  allow_input_downcast=True)

