#%%
# Library Imports
import numpy as np
import fisherMarket as m
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

def pareto_dominated(allocation1, allocation2, valuations, rho):
    # Checks if allocation1 pareto dominates allocation2 under CES
    # utilities and same valuation structure
    utility1 = np.zeros(allocation1.shape[0])
    utility2 = np.zeros(allocation1.shape[0])
    for agent in range(allocation1.shape[0]):
        utility1[agent] = get_ces_utility(allocation1[agent,:], valuations[agent,:], rho)
        utility2[agent] = get_ces_utility(allocation2[agent,:], valuations[agent,:], rho)
    
    return (np.sum((utility1 > utility2)) == allocation1.shape[0])

def is_ir(allocation, weights, valuations, rho):
    # Checks if allocation is individually rational
    ir_is_violated = 0
    for agent in range(weights.shape[0]):
        # Allocation's utility
        utility = get_ces_utility(allocation[agent,:], valuations[agent,:], rho)
        # Utility required by Individual rationality
        equal_alloc = np.repeat(weights[agent], allocation.shape[1])
        min_utility = get_ces_utility(equal_alloc, valuations[agent,:], rho)
        ir_is_violated += (utility>= min_utility)
    
    return (ir_is_violated > 0)

# %%
# Function that runs the DRF mechanism for CES utilities
def run_drf(num_goods, num_agents, valuations, rho, supply, weights, epsilon = 0.001, prices = np.random.rand(num_goods)):
    allocation = np.zeros((num_agents, num_goods))
    demand = np.zeros(num_goods)

    iter = 0
    while (np.sum(np.abs(supply - np.sum(allocation, 0))) > 0.005 and iter < 10000):
        iter += 1
        for good in range(num_goods):
            if (supply[good] < demand[good]):
                prices[good] = np.inf

        for agent in range(num_agents):
            hiksian_demand = get_ces_hicksian(valuations[agent,:], prices, 1 , rho)
            relative_share = np.sum((prices*hiksian_demand)/(weights[agent]))
            demand_agent = (demand < supply)*hiksian_demand*(1/relative_share)*epsilon
            allocation[agent,:] += demand_agent

        demand = np.sum(allocation, 0)
        
    return allocation


# %%
# Set parameters
## Experiment parameters
num_experiments = 20
## Market Parameters
num_goods = 5
num_agents = 3
rho = 0.5
epsilon = 0.001
supply = np.array([1, 1, 1, 1, 1])
weights = np.array([1/2, 1/4, 1/4])

# %%

# Run Analysis of Different results


pareto_dominates_num = 0

for i in range(num_experiments):
    print(f"--- Experiment {i} ---\n")
    # Create Market
    valuations = np.random.rand(num_agents, num_goods)
    market = m.FisherMarket(valuations, weights)
    
    # Solve for market prices and allocations for desired utility function structure.
    alloc_fm, p = market.solveMarket("ces", printResults=False)
    alloc_drf = run_drf(num_goods, num_agents, valuations, rho, supply, weights, epsilon = 0.001, prices = 1/supply)    
    
    is_pd = pareto_dominated(alloc_fm, alloc_drf, valuations, rho)
    num_pareto_dom += is_pd
print(f"Percentage of CE allocations dominating {num_pareto_dom/num_experiments}")

#%%
# Check for IR being respected

num_not_IR = 0

for i in range(num_experiments):
    print(f"--- Experiment {i} ---\n")
    valuation = np.random.rand(num_agents,num_goods)
    alloc = run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 0.001, prices = 1/supply)
    num_not_IR += is_ir(alloc, weights, valuation, rho)

print(f"The percentage of allocations not respecting IR is {num_not_IR/num_experiments}")

# %%
# Check for Incentive-Compatibility Violation

violates_ic = 0
for model in range(10):
    print(f"--- Model {model}---\n")
    valuation = np.random.rand(num_agents, num_goods)
    alloc1 = run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 0.001, prices = 1/supply)
    misreport = np.copy(valuation)
    for expriment in range(100):
        print(f"--- Experiment {expriment}---\n")
        misreport[0,:] = np.random.rand(num_goods)
        alloc2 = run_drf(num_goods, num_agents, misreport, rho, supply, weights, epsilon = 0.001, prices = 1/supply)
        is_not_ic = (get_ces_utility(alloc2[0,:], valuation[0,:], rho) > get_ces_utility(alloc1[0,:], valuation[0,:], rho))
        if(is_not_ic):
            print(f"Fake Utility {get_ces_utility(alloc2[0,:], valuation[0,:], rho)}")                   
            print(f"Real Utility {get_ces_utility(alloc1[0,:], valuation[0,:], rho)}")
        violates_ic += is_not_ic
        print(violates_ic)
print(f"The percentage of allocations not respecting IC is {violates_ic/len(valuations_list)}")

# %%
# Check if there are any allocations not respecting envy-freeness
num_violates_ef = 0
total = 0
for i in range(500):
    total += 1
    print(f"--- Experiment {i} ---")
    valuation = np.random.rand(num_agents, num_goods)
    alloc = run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 0.001, prices = 1/supply)
    for agent1 in range(num_agents):
        for agent2 in range(num_agents):
            if (agent1 != agent2):
                alloc1 = alloc[agent1,:]
                alloc2 = alloc[agent2,:]
                is_not_ef = (1/weights[agent2])*(get_ces_utility(alloc2, valuation[agent1,:], rho)) > (1/weights[agent1])*(get_ces_utility(alloc1, valuation[agent1,:], rho))
                if(is_not_ef):
                    print(f"Other utility {(1/weights[agent2])*(get_ces_utility(alloc2, valuation[agent1,:], rho))}")
                    print(f"Real Utility {(1/weights[agent1])*(get_ces_utility(alloc1, valuation[agent1,:], rho))}")
                num_violates_ef += is_not_ef


print(f"The percentage of allocations not respecting EF is {num_violates_ef/total}")

# %%
