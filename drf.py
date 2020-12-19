#%%
# Library Imports
import numpy as np
import pickle
np.set_printoptions(precision=2)

# Economic Function Definitions

def get_ces_expend(valuations, prices, util,rho):
    # Returns the expenditure for CES utility
    return util/(np.sum((1/np.power(prices, rho))*valuations)**(1/rho))

def get_ces_hicksian(valuations, prices, util ,rho):
    # Returns the hicksian demand for CES utilities
    return np.power((1/prices)*valuations, 1/(1-rho))*util/(np.sum((1/np.power(prices, rho))*valuations)**(1/(rho*(1-rho))))

def get_ces_utility(allocations, valuations, rho):
    # Returns utility value for CES agent
    return (np.power(allocations, rho).T @ valuations)**(1/rho)

def get_ces_welfare(allocations, valuations, rho):
    # Returns utilitarian welfare
    welfare = 0
    for agent in range(allocations.shape[0]):
        welfare += get_ces_utility(allocations[agent,:], valuations[agent,:], rho)
    return welfare

def pareto_dominates(allocation1, allocation2, valuations, rho):
    # Checks if one of the allocations pareto dominates the other under CES
    # utilities and same valuation structure
    utility1 = np.zeros(allocation1.shape[0])
    utility2 = np.zeros(allocation1.shape[0])
    for agent in range(allocation1.shape[0]):
        utility1[agent] = get_ces_utility(allocation1[agent,:], valuations[agent,:], rho)
        utility2[agent] = get_ces_utility(allocation2[agent,:], valuations[agent,:], rho)
    
    return (np.sum((utility1 > utility2)) == allocation1.shape[0]) or (np.sum((utility1 < utility2)) == allocation1.shape[0])

#%%
# Set parameters
## Experiment parameters
num_runs = 10000
## Market Parameters
num_goods = 5
num_agents = 3
valuations = np.random.rand(num_agents, num_goods)
rho = -0.5
epsilon = 0.001
supply = np.array([3, 4, 5, 2, 1])


# %%
# Function that runs the DRF mechanism for CES utilities
def run_drf(num_goods, num_agents, valuations, rho, supply, epsilon = 0.001, prices = np.random.rand(num_goods)):
    allocation = np.zeros((num_agents, num_goods))
    demand = np.zeros(num_goods)

    iter = 0
    while (np.sum(np.abs(supply - np.sum(allocation, 0))) > 0.005 and iter < 10000):
        iter += 1
        for good in range(num_goods):
            if (supply[good] < demand[good]):
                prices[good] = 10000000

        for agent in range(num_agents):
            demand_agent = get_ces_hicksian(valuations[agent,:], prices, 1 , rho)
            demand_agent = (demand < supply)*demand_agent*(1/np.sum(demand_agent))*epsilon
            allocation[agent,:] += demand_agent

        demand = np.sum(allocation, 0)
        
        if (iter % 1000 ==0):
            print(f"Iteration {iter} - Demand: {demand}")
    
    return allocation


# %%
alloc_list = []

# Run experiments

for t in range(num_runs):
    print(f"---Experiment {t+1}---\\")
    alloc_list.append(run_drf(num_goods, num_agents, valuations, rho, supply, epsilon = 0.001))

# %%
# Check if any allocation pareto dominates the others
# Variable to store number of allocations that are pareto dominated
pareto_dominates_num = 0
for i in range(num_runs-1):
    for j in range(i+1, num_runs):
        print(f"Allocation {i} and Allocation {j}:\n")
        is_pd = pareto_dominates(alloc_list[i], alloc_list[j], valuations, rho)
        print(f"Is pareto dominated?: {is_pd}")
        pareto_dominates_num += is_pd
print(f"There exists {pareto_dominates_num} allocations that pareto-dominate")


# %%
alloc = run_drf(num_goods, num_agents, valuations, rho, supply, epsilon = 0.001, prices = 1/supply)
pareto_dominates_num = 0
for i in range(num_runs):
        print(f"Allocation {i}:\n")
        is_pd = pareto_dominates(alloc_list[i], alloc, valuations, rho)
        print(f"Is pareto dominated?: {is_pd}")
        pareto_dominates_num += is_pd
print(f"There exists {pareto_dominates_num} allocations that pareto-dominate")

# %%
with open("test.txt", "wb") as fp:   #Pickling
    pickle.dump(alloc_list, fp)

with open("test.txt", "rb") as fp:   # Unpickling
    alloc_list = pickle.load(fp)
# %%
