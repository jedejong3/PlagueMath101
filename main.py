import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import EoN
import random
import numpy as np
import ffmpeg
import scipy
import scipy.stats
from scipy.stats import skewnorm

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15)

class graph_and_summary:
    #G is the full graph, and H just shows the powerlaw connecting counties
    def __init__(self, G, H):
        self.G=G
        self.H=H

def get_seed_multidigraph(c):
    ''' Returns a complete digraph on c+1 vertices'''
    graph = nx.gnp_random_graph(c+1,p=1,directed=True)
    seed_graph = nx.MultiDiGraph()
    seed_graph.add_nodes_from(graph)
    seed_graph.add_edges_from(graph.edges())

    return seed_graph

def get_skew_distribution(a, num_vertices):
    # A is the skew, and num_vertices tells how many samples to take
    r = skewnorm.rvs(a, size=num_vertices)
    lower, upper = skewnorm.interval(.999, a)
    # shifted = [x - lower for x in r]
    diff = upper - lower
    rescaled = [abs((x-lower)/diff)/2 for x in r]
    return rescaled


# def get_safety(avg_safety):
#     scipy.stats.skewnorm()
#     (np.random.normal(0, 1, 1) + 3) / 6
#     return abs(np.random.normal(avg_safety, avg_safety/3))

def get_fitness(sigma):
    ''' Samples from the standard lognormal distribution with std dev = sigma'''
    return np.random.lognormal(0, sigma)


def get_random_node(G):
    ''' Returns a random node from the graph G '''
    return random.choice(list(G.nodes))


def build_graph(nws_n, nws_k, nws_p, lnfa_c, lnfa_sigma, lnfa_num_steps):#, a):
    G = nx.Graph()
    H_summary = nx.Graph()
    Gprefix = ""
    # hs stores all the subgraphs
    hs = []
    # fitness list will store the fitness for each of the nodes
    fitness_list = []
    # an empty list to store the safety values for any given node
    # safety_dist = []
    for n in range(lnfa_num_steps):
        H = nx.newman_watts_strogatz_graph(nws_n, nws_k, nws_p)
        H = nx.relabel_nodes(H, lambda x: x + n*nws_n)
        G = nx.union(G, H)
        H_summary.add_node(n)
        if n > 0:
            to_add_index = random.choices(range(n), weights=fitness_list, k=lnfa_c)
            # to_add = hs[to_add_index]
            for j in range(lnfa_c):
                to_add = hs[to_add_index[j]]
                G.add_edge(get_random_node(H), get_random_node(to_add))
                H_summary.add_edge(n, to_add_index[j])
        hs.append(H)
        fitness_list.append(get_fitness(lnfa_sigma))
        #ToDo: vary the number of nodes by setting them to follow the distribution
        # set up for the fitness, normalized to the total number of nodes, and then assign fitness to that number

        # if n == lnfa_num_steps-1:
        #     a = -4
        # print(a)
        # safety_dist = safety_dist + get_skew_distribution(a, nws_n)
    # safety_dist = get_skew_distribution(a, (lnfa_num_steps*nws_n))
    # the safety needs to be zipped into a dictionary to apply it to each vertex
    # safety = {i: safety_dist[i] for i in range(lnfa_num_steps*nws_n)}

    # Code for plotting the safety distribution for the graph
    # y = safety_dist
    # plt.hist(y, bins=100)
    # plt.gca().set(title='Safety Distribution: Skew = ' + str(a), ylabel='Frequency');
    # plt.show()


    # nx.set_node_attributes(G, safety, 'safety')
    return G,H_summary


def assign_safety(G, county_safety_list, county_size_list):
    safety_dist = []
    for i in range(len(county_safety_list)):
        safety_dist = safety_dist + get_skew_distribution(county_safety_list[i], county_size_list[i])
    # y = safety_dist
    # plt.hist(y, bins=100)
    # plt.gca().set(title='Safety Distribution: Skew = ' + str(county_safety_list[i]), ylabel='Frequency', xlabel='Rate of Transmission');
    # plt.savefig('safety_skew' + str(county_safety_list[i]))
    #
    # plt.show()
    safety = {i: safety_dist[i] for i in range(sum(county_size_list))}
    nx.set_node_attributes(G, safety, 'safety')
    return G



######################################################################################################################
# Code for setting up to compare different safety levels


# for avg_safety in [-4, 0, 4]:
#     G = build_graph(smallworld_size, smallworld_k_nearest_neighbors, smallworld_odds_of_rewire, powerlaw_num_connections, powerlaw_sigma, powerlaw_num_nodes, avg_safety)

# smallworld_size = 100
# smallworld_k_nearest_neighbors = 4
# smallworld_odds_of_rewire = .2
# powerlaw_num_connections = 0
# powerlaw_sigma = 0
# powerlaw_num_nodes = 1
# avg_safety = -4

############################################################################################################
# Code to build a smallworld network to show the spread on a small scale
# G1 = build_graph(smallworld_size, smallworld_k_nearest_neighbors, smallworld_odds_of_rewire, powerlaw_num_connections,
#                 powerlaw_sigma, powerlaw_num_nodes, avg_safety)
# nx.draw_circular(G1)
#
# plt.show()
#
# weight_sum = 0
# inv_weight_sum = 0
#
#
# for edge in G1.edges():
#     G1.edges[edge[0],edge[1]]['weight'] = G1.nodes[edge[0]]['safety']*G1.nodes[edge[1]]['safety']
#     weight_sum += G1.nodes[edge[0]]['safety']*G1.nodes[edge[1]]['safety']
#
# tmax = 20
# iterations = 5  #run 5 simulations
# tau = 0.1           #transmission rate
# gamma = 1.0    #recovery rate
# rho = 0.05      #random fraction initially infected
#
# sim = EoN.fast_SIR(G1, G1.number_of_edges()/weight_sum, gamma, rho=rho, transmission_weight= 'weight', tmax = tmax, return_full_data=True)
#
#
# ani = sim.animate(ts_plots=['SI', 'SIR'])
# filename = "smallworld_animation_100.mp4"
# ani.save(filename, extra_args=['-vcodec', 'libx264'])




##########################################################################################################
# Code to pick out the summaries for each of the midsized components
smallworld_size = 50
smallworld_k_nearest_neighbors = 4
smallworld_odds_of_rewire = .2
powerlaw_num_connections = 2
powerlaw_sigma = 4
powerlaw_num_nodes = 30
# avg_safety = 10
# smallworld_sizes = [10, 20, 50, 100]
# powerlaw_num_nodess = [10, 20, 30, 40, 50]
smallworld_sizes = [10]
powerlaw_num_nodess = [10]
for smallworld_size in smallworld_sizes:
    for powerlaw_num_nodes in powerlaw_num_nodess:

        G, H_summary = build_graph(smallworld_size, smallworld_k_nearest_neighbors, smallworld_odds_of_rewire, powerlaw_num_connections,
                            powerlaw_sigma, powerlaw_num_nodes)
        county_size_list = np.ones(powerlaw_num_nodes, dtype=int)*smallworld_size



        safety_generator_list = [-10,-4, 0, 4,10]
        plt.figure(figsize=(6, 15))
        plt.subplots_adjust(hspace=.5)
        plt.subplot(len(safety_generator_list)+1, 1, 1)
        color_map = plt.get_cmap('gist_rainbow')
        colors = color_map(list(i/powerlaw_num_nodes for i in range(powerlaw_num_nodes)))
        nx.draw(H_summary, node_color=colors, node_size=50)
        plt.suptitle("Community size = " + str(smallworld_size) + "\n Powerlaw size = " +str(powerlaw_num_nodes))
        for s in range(len(safety_generator_list)):
            avg_safety = safety_generator_list[s]
            safety_list = np.ones(powerlaw_num_nodes, dtype=int)*avg_safety
            assign_safety(G, safety_list, county_size_list)


            weight_sum = 0
            inv_weight_sum = 0


            for edge in G.edges():
                G.edges[edge[0],edge[1]]['weight'] = G.nodes[edge[0]]['safety']*G.nodes[edge[1]]['safety']
                weight_sum += G.nodes[edge[0]]['safety']*G.nodes[edge[1]]['safety']

            tmax = 20
            iterations = 1  #run 5 simulations
            tau = 0.1           #transmission rate
            gamma = 1.0    #recovery rate
            rho = 0.01      #random fraction initially infected


            sim = EoN.fast_SIR(G, G.number_of_edges()/weight_sum, gamma, rho=rho, transmission_weight= 'weight', tmax = tmax, return_full_data=True)
            # ani = sim.animate(ts_plots=['I', 'SIR'])
            # filename = "smallworld_animation_for_comparison.mp4"
            # ani.save(filename, extra_args=['-vcodec', 'libx264'])
            # plt.figure()
            # plt.suptitle('Safety Skew ' + str(avg_safety))

            plt.subplot(len(safety_generator_list)+1, 2, 3+2*s)
            plt.title('Total Infections; Safety = ' + str(avg_safety))
            plt.xlabel('$t$')
            # plt.ylabel('Number infected')
            plt.plot(sim.t(), sim.I(), alpha=0.3, color='r', label='Total Infection S = '+str(avg_safety))

            # for t in range(20):
            #     print(sim.get_statuses(nodelist=range(smallworld_size), time=t))

            timestamps = sim.summary()[0]

            summs = []
            infection_breakdown = []
            for h in range(powerlaw_num_nodes):
                summs.append(sim.summary(nodelist=range(h*smallworld_size, (h+1)*smallworld_size-1)))
                individual_summary = []
                infection_breakdown.append(individual_summary)

            for t in timestamps:
                for i in range(len(summs)):
                    summ = summs[i]
                    if t < list(summ[0])[-1]:
                        res = list(filter(lambda i: i > t, summ[0]))[0]
                        num_infected = summ[1].get('I')[list(summ[0]).index(res)-1]
                    else:
                        num_infected = summ[1].get('I')[-1]
                    # print("Community " + str(i) + " has " + str(num_infected) + " infections")
                    infection_breakdown[i].append(num_infected)

            # print(infection_breakdown)
            # print(np.array(infection_breakdown))

            x = timestamps
            plt.subplot(len(safety_generator_list)+1, 2, 4 + 2*s)
            plt.xlabel('$t$')
            # plt.ylabel('Number infected')


            for h in range(powerlaw_num_nodes):
                y = infection_breakdown[h]
                plt.plot(x, y, alpha=0.3, label='County ' + str(h+1), color=colors[h])

            plt.title('County Infections; Safety = ' + str(avg_safety))
            outfile_template = 'small_world_lnfa-{0}-{1}-{2}.gexf'
            outfile = outfile_template.format(smallworld_size, powerlaw_num_nodes, avg_safety)
            nx.write_gexf(G, outfile)
        # plt.savefig("Community size = " + str(smallworld_size) + ", Powerlaw size = " +str(powerlaw_num_nodes) + str(2))
        plt.show()



        # print('writing to ' + outfile)

        # nx.write_gexf(nx.newman_watts_strogatz_graph(smallworld_size, smallworld_k_nearest_neighbors, smallworld_odds_of_rewire), "smallworld.gexf")
        # nx.write_gexf(H_summary, outfile)


#########################################################################################################
# The rest of the code for comparative safety

# smallworld_size = 20
# smallworld_k_nearest_neighbors = 4
# smallworld_odds_of_rewire = .2
# powerlaw_num_connections = 1
# powerlaw_sigma = 8
# powerlaw_num_nodes = 100
# avg_safety = -10
# G = build_graph(smallworld_size, smallworld_k_nearest_neighbors, smallworld_odds_of_rewire, powerlaw_num_connections,
#                 powerlaw_sigma, powerlaw_num_nodes, avg_safety)
#
#
# weight_sum = 0
# inv_weight_sum = 0
#
#
# for edge in G.edges():
#     G.edges[edge[0],edge[1]]['weight'] = G.nodes[edge[0]]['safety']*G.nodes[edge[1]]['safety']
#     weight_sum += G.nodes[edge[0]]['safety']*G.nodes[edge[1]]['safety']
#
# tmax = 20
# iterations = 5  #run 5 simulations
# tau = 0.1           #transmission rate
# gamma = 1.0    #recovery rate
# rho = 0.005      #random fraction initially infected
#
#
# for counter in range(iterations): #run simulations
#     t, S, I, R = EoN.fast_SIR(G, G.number_of_edges()/weight_sum, gamma, rho=rho, transmission_weight= 'weight', tmax = tmax)
#     if counter == 0:
#         plt.plot(t, I, color = 'r', alpha=0.3, label='less cautious')
#     plt.plot(t, I, color = 'r', alpha=0.3)
#
# avg_safety = 0
# G = build_graph(smallworld_size, smallworld_k_nearest_neighbors, smallworld_odds_of_rewire, powerlaw_num_connections,
#                 powerlaw_sigma, powerlaw_num_nodes, avg_safety)
#
# weight_sum = 0
# inv_weight_sum = 0
#
#
# for edge in G.edges():
#     G.edges[edge[0],edge[1]]['weight'] = G.nodes[edge[0]]['safety']*G.nodes[edge[1]]['safety']
#     weight_sum += G.nodes[edge[0]]['safety']*G.nodes[edge[1]]['safety']
#
#
# for counter in range(iterations): #run simulations
#     t, S, I, R = EoN.fast_SIR(G, G.number_of_edges()/weight_sum, gamma, rho=rho, transmission_weight= 'weight', tmax = tmax)
#     if counter == 0:
#         plt.plot(t, I, color = 'k', alpha=0.3, label='average')
#     plt.plot(t, I, color = 'k', alpha=0.3)
#
# avg_safety = 10
# G = build_graph(smallworld_size, smallworld_k_nearest_neighbors, smallworld_odds_of_rewire, powerlaw_num_connections,
#                 powerlaw_sigma, powerlaw_num_nodes, avg_safety)
#
# weight_sum = 0
# inv_weight_sum = 0
#
#
# for edge in G.edges():
#     G.edges[edge[0],edge[1]]['weight'] = G.nodes[edge[0]]['safety']*G.nodes[edge[1]]['safety']
#     weight_sum += G.nodes[edge[0]]['safety']*G.nodes[edge[1]]['safety']
#
# for counter in range(iterations): #run simulations
#     t, S, I, R = EoN.fast_SIR(G, G.number_of_edges()/weight_sum, gamma, rho=rho, transmission_weight= 'weight', tmax = tmax)
#     if counter == 0:
#         plt.plot(t, I, color = 'b', alpha=0.3, label='more cautious')
#     plt.plot(t, I, color = 'b', alpha=0.3)




#########################################################################################################################




# # # Variable but constant tau
# # tau = .2
# # for counter in range(iterations):  # run simulations
# #     t, S, I, R = EoN.fast_SIR(G, tau, gamma, rho=rho, transmission_weight= 'weight', tmax=tmax)
# #     if counter == 0:
# #         plt.plot(t, I, color='b', alpha=0.3, label='Simulation t = .2')
# #     plt.plot(t, I, color='b', alpha=0.3)
# #
# # tau = .3
# # for counter in range(iterations):  # run simulations
# #     t, S, I, R = EoN.fast_SIR(G, tau, gamma, rho=rho, transmission_weight= 'weight', tmax=tmax)
# #     if counter == 0:
# #         plt.plot(t, I, color='r', alpha=0.3, label='Simulation t = .3')
# #     plt.plot(t, I, color='r', alpha=0.3)
#

# plt.xlabel('$t$')
# plt.ylabel('Number infected')
#
# plt.legend()
# plt.savefig('SIR_BA_model_vs_sim.png')
# plt.show()