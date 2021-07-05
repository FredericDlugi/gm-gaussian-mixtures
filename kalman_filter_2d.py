from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.gaussian_mixture_reduction import reduction_algorithm
from ime_fgs.messages import GaussianMixtureMeanCovMessage, GaussianMixtureWeightedMeanInfoMessage
import matplotlib.pyplot as plt
import numpy as np

# State estimation with a Kalman filter, using the ime-fgs toolbox
# This is a simple example to introduce the toolbox, for more serious uses please take a look at the Kalman class
#
# Single time slice of a Kalman filter:
#
#
#               process_noise_in_node
#                     +---+
#                     |   |
#                     +---+
#                       |
#                       | I_k
#                       v             equality_node
#  X_k +---+  X''_k   +---+     X_k+1     +---+ X'_k+1
# ---->| A |--------->| + |-------------->| = |------->
#      +---+          +---+               +---+
#      A_node   add_process_noise_node      |
#                                           | X''_k+1
#                                           v
#                                         +---+
#                                         | C | C_node
#                                         +---+
#                                           |
#                                           | Y_k+1
#                                           v
#                             +---+ D_k+1 +---+
#                             |   |------>| + | add_meas_noise_node
#                             +---+       +---+
#                       meas_noise_node     |
#                                           | Z_k+1
#                                           v
#                                          +-+
#                                          +-+
#                                       meas_in_node
#


# System parameters
A = [[1, 0.1, 0, 0], [-0.01, 0.99, 0, 0], [0, 0, 1, 0.1], [0, 0, -0.01, 0.99]]  # adjust naming convention for the slice above
C = [[1, 0, 0, 0],[0, 0, 1, 0]]  # only the first state was measured
meas_noise_cov = [[0.5, 0], [0, 0.5]]  # the measurement noise
process_noise_cov = [[0.01, 0.003, 0, 0], [0, 0.01, 0, 0], [0, 0, 0.01, 0.003], [0, 0, 0, 0.01]]  # we assume no noise for the process

# read measurements of of the first state of the system
measurements = np.load("measurements_2d.npy")
# read true states of the system (as reference)
true_states = np.load("true_states_2d.npy")
# read time vector corresponding to measurements and states
t_vec = np.load("t_vec_2d.npy")



meas_in_list_msg = []
for i in range(measurements.shape[0]):
    meas_in_list_msg += [GaussianMixtureMeanCovMessage([[1]], [measurements[i,:].reshape(2,1)], [[[0,0],[0,0]]]) ]

# Create all relevant nodes for a single Kalman slice
# including an additional PriorNode for the state input and state output, which act as a terminator
# The name is optional, but may come handy if you are trying to debug your graph
state_in_node = PriorNode(name="x_in")
state_out_node = PriorNode(name="x_out")
A_node = MatrixNode(A, name="A")
C_node = MatrixNode(C, name="C")
meas_noise_node = PriorNode(GaussianMixtureMeanCovMessage([[1]], [[[0],[0]]], [meas_noise_cov]), name="N_D")
equality_node = EqualityNode(name="=")

add_process_noise_node = AdditionNode(name="add_process_noise_node")
add_meas_noise_node = AdditionNode(name="add_meas_noise_node")
process_noise_in_node = PriorNode(
    GaussianMixtureMeanCovMessage([[1]], [[[0], [0], [0], [0]]], [process_noise_cov]), name="process_noise_in_node"
)
meas_in_node = PriorNode(name="meas_in_node")

# Connect the nodes together with the .connect function
equality_node.ports[1].connect(C_node.port_a)
equality_node.ports[2].connect(state_out_node.port_a)

C_node.port_b.connect(add_meas_noise_node.port_a)


state_in_node.port_a.connect(A_node.port_a)
A_node.port_b.connect(add_process_noise_node.port_a)

process_noise_in_node.port_a.connect(add_process_noise_node.port_b)
add_process_noise_node.port_c.connect(equality_node.ports[0])

meas_noise_node.port_a.connect(add_meas_noise_node.port_b)
add_meas_noise_node.port_c.connect(meas_in_node.port_a)
# set a (wrong) start state with high variances (low confidence)
start_state = GaussianMixtureMeanCovMessage([[1]], [[[0], [0.1], [5], [0.1]]], [[[99, 99, 0, 0], [0, 99, 0, 0], [0, 0, 99, 99], [0, 0, 0, 99]]])

# Create a list of estimated messages by updating all ports for the Kalman filter for every time step
estimated_state_list = [start_state]
for idx, t in enumerate(t_vec):
    # use last state estimation for new state estimation
    state_in_node.update_prior(estimated_state_list[-1])
    meas_in_node.update_prior(meas_in_list_msg[idx])

    A_node.port_b.update(GaussianMixtureMeanCovMessage)
    add_process_noise_node.port_c.update(GaussianMixtureWeightedMeanInfoMessage)
    add_meas_noise_node.port_a.update(GaussianMixtureWeightedMeanInfoMessage)
    C_node.port_a.update(GaussianMixtureWeightedMeanInfoMessage)
    # append new estimated state to list
    estimated_state_list.append(reduction_algorithm(equality_node.ports[2].update(GaussianMixtureMeanCovMessage), 3))

# Plot result
# Extract results
estimated_position_list = [x.mean[0, (0, 2)] for x in estimated_state_list]
estimated_speed_list = [x.mean[0,(1, 3)] for x in estimated_state_list]
estimated_positions = np.array(estimated_position_list)
estimated_speed = np.array(estimated_speed_list)

# Use two subplots to plot the position and the speed of the mass
ax1 = plt.subplot(2, 1, 1)
ax1.scatter(measurements[:, 0], measurements[:, 1], label="simulated measurement")
ax1.scatter(true_states[:, 0], true_states[:, 2], label="true state")
ax1.scatter(estimated_positions[:, 0], estimated_positions[:, 1], label="estimated state")
ax1.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(t_vec, true_states[:, 1], label="true state X")
ax2.plot(t_vec, true_states[:, 3], label="true state Y")
ax2.plot(t_vec, estimated_speed[:-1, 0], label="estimated state X")
ax2.plot(t_vec, estimated_speed[:-1, 1], label="estimated state Y")
ax2.legend()

plt.suptitle("Mass Spring Damper System")
ax1.set(ylabel="position p in m")
ax2.set(xlabel="position p in m", ylabel="position p in m")

ax1.grid()
ax2.grid()

plt.show()
