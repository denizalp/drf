#%%
# Library Imports
import numpy as np
import fisherMarket as m
import pickle
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

# Economic Function Definitions
def get_ces_utility(allocations, valuations, rho):
    # Returns utility value for CES agent
    return np.power(np.power(allocations, rho).T @ valuations, (1/rho))

def get_ces_value(valuations, prices, budget, rho):
    sigma = 1/(1-rho)
    v = np.power(valuations, sigma)
    p = np.power(prices, 1-sigma)
    cost_unit_util = np.power(v.T @ p, 1/(1-sigma))
    return budget/cost_unit_util

def get_ces_demand(valuations, prices, budget, rho):
    v = np.power(valuations, 1/(1-rho))
    p_num = np.power(prices, 1/(rho-1))
    p_denom = np.power(prices, rho/(rho-1))
    return budget* ((v*p_num)/ (v.T @ p_denom))

def get_ces_expend(valuations, prices, util,rho):
    # Returns the expenditure for CES utility
    sigma = 1/(1-rho)
    v = np.power(valuations, sigma)
    p = np.power(prices, 1-sigma)
    cost_unit_util = np.power(v.T @ p, 1/(1-sigma))
    return util*cost_unit_util


def get_ces_hicksian(valuations, prices, util ,rho):
    # Returns the hicksian demand for CES utilities
    expenditure = get_ces_expend(valuations, prices, util,rho)
    return get_ces_demand(valuations, prices, expenditure, rho)



def get_ces_welfare(allocations, valuations, rho):
    # Returns utilitarian welfare
    welfare = 0
    for agent in range(allocations.shape[0]):
        welfare += get_ces_utility(allocations[agent,:], valuations[agent,:], rho)
    return welfare

def get_leontief_utility(allocations, valuations):
    return np.min(allocations/valuations) 

def get_leontief_value(valuations, prices, budget):
    return budget/(prices.T @ valuations)

def get_leontief_demand(valuations, prices, budget):
    return (budget/(prices.T @ valuations))* valuations
    
def get_leontief_expend(valuations, prices, util):
    return util*(prices.T @ valuations)

def get_leontief_hicksian(valuations, prices, util):
    return valuations*util

def get_leontief_welfare(allocations, valuations):
    # Returns utilitarian welfare
    welfare = 0
    for agent in range(allocations.shape[0]):
        welfare += get_leontief_utility(allocations[agent,:], valuations[agent,:])
    return welfare

def exists_pareto_dominated(allocation1, allocation2, valuations, rho):
    # Checks if one of the allocations pareto dominates the other under CES
    # utilities and same valuation structure
    utility1 = np.zeros(allocation1.shape[0])
    utility2 = np.zeros(allocation1.shape[0])
    for agent in range(allocation1.shape[0]):
        utility1[agent] = get_ces_utility(allocation1[agent,:], valuations[agent,:], rho)
        utility2[agent] = get_ces_utility(allocation2[agent,:], valuations[agent,:], rho)
    
    return (np.sum((utility1 > utility2)) == allocation1.shape[0]) or (np.sum((utility1 < utility2)) == allocation1.shape[0])

def pareto_dominates(allocation1, allocation2, valuations, rho):
    # Checks if allocation1 pareto dominates allocation2 under CES
    # utilities and same valuation structure
    utility1 = np.zeros(allocation1.shape[0])
    utility2 = np.zeros(allocation1.shape[0])
    
    for agent in range(allocation1.shape[0]):
        utility1[agent] = get_ces_utility(allocation1[agent,:], valuations[agent,:], rho)
        utility2[agent] = get_ces_utility(allocation2[agent,:], valuations[agent,:], rho)

    return (np.sum((utility1 >= utility2)) == allocation1.shape[0]) and (np.sum((utility1 > utility2)) > 0)

def is_not_ir(allocation, weights, valuations, rho):
    # Checks if allocation is individually rational
    ir_is_violated = 0
    for agent in range(weights.shape[0]):
        # Allocation's utility
        utility = get_ces_utility(allocation[agent,:], valuations[agent,:], rho)
        # Utility required by Individual rationality
        equal_alloc = np.repeat(weights[agent], allocation.shape[1])
        min_utility = get_ces_utility(equal_alloc, valuations[agent,:], rho)
        ir_is_violated += (utility < min_utility)
    
    return (ir_is_violated > 0)

# %%
# Function that runs the DRF mechanism for CES utilities
def run_drf(num_goods, num_agents, valuations, rho, supply, weights, prices, epsilon = 0.01):
    allocation = np.zeros((num_agents, num_goods))
    demand = np.zeros(num_goods)
    is_saturated = {good:False for good in range(num_goods)}
    t = 0

    while (np.sum(np.abs(supply - demand)) > 0.005 and t < 2000):
        t += 1
        demand = np.sum(allocation, 0)

        for good in range(num_goods):
            if (supply[good] <= demand[good] and not is_saturated[good]):
                prices = 1/(supply-demand)
                is_saturated[good] = True
                # print(good)

            if(supply[good] <= demand[good]):
                prices[good] = np.inf
 
        for agent in range(num_agents):
            hiksian_demand = np.nan_to_num(get_ces_hicksian(valuations[agent,:], prices, 1 , rho))
            dominant_share = 0
            for good in range(num_goods):
                share = hiksian_demand[good]*prices[good]
                if (not is_saturated[good]) and (share > dominant_share):
                    dominant_share = share
            demand_agent = epsilon*(np.nan_to_num(weights[agent]/dominant_share))*hiksian_demand
            allocation[agent,:] += ((demand + demand_agent)<supply)*demand_agent
    
    demand = np.sum(allocation, 0)
    if ((demand >= 1.1).any()):
        print(demand)
    assert (demand <= 1.1).all() 
    return allocation


# %%
# Run Analysis of Different results
# Set parameters
## Experiment parameters
num_experiments = 20
## Market Parameters
num_goods = 5
num_agents = 3
rho = 0.5
epsilon = 0.001
supply = np.array([1, 1, 1, 1, 1])
weights = np.array([1/3, 1/3, 1/3])
valuations = np.random.rand(num_agents, num_goods)

# Test
X = run_drf(num_goods, num_agents, valuations, rho, supply, weights, epsilon = 0.005, prices = 1/supply)
print(np.sum(X, axis = 0))

# %%

# Compare to competitive equilibrium


ce_dominates = 0
drf_dominates = 0

ce_welfare = []
drf_welfare =[]
for i in range(500):
    if(not (i % 50)):
        print(f"--- Experiment {i} ---\n")
    # Create Market
    valuations = np.random.rand(num_agents, num_goods)
    market = m.FisherMarket(valuations, weights)
    
    # Solve for market prices and allocations for desired utility function structure.
    alloc_fm, p = market.solveMarket("ces", printResults=False)
    alloc_drf = run_drf(num_goods, num_agents, valuations, rho, supply, weights, epsilon = 0.005, prices = 1/supply)    
    
    ce_dominates += pareto_dominates(alloc_fm, alloc_drf, valuations, rho)
    drf_dominates += pareto_dominates(alloc_drf, alloc_fm, valuations, rho)
    ce_welfare.append(get_ces_welfare(alloc_fm, valuations, rho))
    drf_welfare.append(get_ces_welfare(alloc_drf, valuations, rho))

print(f"Percentage of CE allocations dominating {ce_dominates/(i+1)}")
print(f"Percentage of drf allocations dominating {drf_dominates/(i+1)}")

plt.figure(figsize=(8,6))
plt.hist(ce_welfare, bins=50, alpha=0.5, label="Competitive Eqm. Welfare")
plt.hist(drf_welfare, bins=50, alpha=0.5, label="DRF Welfare")

plt.xlabel("Welfare", size=14)
plt.ylabel("Count", size=14)
plt.title("Comparing the Welfares of DRF and Competitive Equilibria")
plt.legend(loc='upper right')

#%%
# Check for IR being respected

num_not_IR = 0

for i in range(1000):
    if (not (i % 50) ):
        print(f"--- Experiment {i} ---\n")
    
    valuation = np.random.rand(num_agents,num_goods)
    alloc = run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 0.005, prices = 1/supply)
    alloc_is_not_ir = is_not_ir(alloc, weights, valuation, rho)
    if (alloc_is_not_ir):
        print(np.sum(alloc, axis = 0))
        print("There is an IR violation")
    num_not_IR += alloc_is_not_ir

print(f"The percentage of allocations not respecting IR is {num_not_IR/(i+1)}")

# %%
# Check for Incentive-Compatibility Violation

violates_ic = 0
for model in range(100):
    print(f"--- Model {model}---\n")
    valuation = np.random.rand(num_agents, num_goods)
    alloc1 = run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 0.005, prices = 1/supply)
    misreport = np.copy(valuation)
    for experiment in range(1000):
        if(not (experiment % 50)):
            print(f"--- Experiment {experiment}---\n")
        misreport[0,:] = np.random.rand(num_goods)
        alloc2 = run_drf(num_goods, num_agents, misreport, rho, supply, weights, epsilon = 0.005, prices = 1/supply)
        is_not_ic = (get_ces_utility(alloc2[0,:], valuation[0,:], rho) > get_ces_utility(alloc1[0,:], valuation[0,:], rho))
        if(is_not_ic):
            violates_ic += is_not_ic
            print(f"Fake Utility {get_ces_utility(alloc2[0,:], valuation[0,:], rho)}")                   
            print(f"Real Utility {get_ces_utility(alloc1[0,:], valuation[0,:], rho)}")
            break


        


print(f"The percentage of allocations not respecting IC is {violates_ic/(model+1)}")

# %%
# Check if there are any allocations not respecting envy-freeness
num_violates_ef = 0
total = 0
for i in range(50):
    total += 1
    if(not (i % 50)):
        print(f"--- Experiment {i} ---")
    valuation = np.random.rand(num_agents, num_goods)
    alloc = run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 1, prices = 1/supply)
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

#%%
print("-----True Valuations------")
run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 1, prices = 1/supply)

print("---------Misreport--------")
run_drf(num_goods, num_agents, misreport, rho, supply, weights, epsilon = 1, prices = 1/supply)
# %%

# Do everything for leontief

def pareto_dominates(allocation1, allocation2, valuations):
    # Checks if allocation1 pareto dominates allocation2 under CES
    # utilities and same valuation structure
    utility1 = np.zeros(allocation1.shape[0])
    utility2 = np.zeros(allocation1.shape[0])
    
    for agent in range(allocation1.shape[0]):
        utility1[agent] = get_leontief_utility(allocation1[agent,:], valuations[agent,:])
        # print(f"Allocation1 : Agent {agent} - Utility {utility1[agent]}")
        utility2[agent] = get_leontief_utility(allocation2[agent,:], valuations[agent,:])
        # print(f"Allocation2 : Agent {agent} - Utility {utility2[agent]}")

    return (np.sum((utility1 >= utility2)) == allocation1.shape[0]) and (np.sum((utility1 > utility2)) > 0)

def is_not_ir(allocation, weights, valuations, rho):
    # Checks if allocation is individually rational
    ir_is_violated = 0
    for agent in range(weights.shape[0]):
        # Allocation's utility
        utility = get_leontief_utility(allocation[agent,:], valuations[agent,:])
        # Utility required by Individual rationality
        equal_alloc = np.repeat(weights[agent], allocation.shape[1])
        min_utility = get_leontief_utility(equal_alloc, valuations[agent,:])
        ir_is_violated += (utility < min_utility)
    
    return (ir_is_violated > 0)

# Function that runs the DRF mechanism for CES utilities
def run_drf(num_goods, num_agents, valuations, rho, supply, weights, prices, epsilon = 0.01):
    allocation = np.zeros((num_agents, num_goods))
    demand = np.zeros(num_goods)

    t = 0
    while (np.sum(np.abs(supply - demand)) > 0.005 and t < 1000):
        t += 1
        demand = np.sum(allocation, 0)
        prices = 1/(supply-demand)

        for good in range(num_goods):
            if (supply[good] <= demand[good]):
                prices[good] = np.inf
        
        for agent in range(num_agents):
            hiksian_demand = np.nan_to_num(get_leontief_hicksian(valuations[agent,:], prices, 1 ))
            expenditure =  np.sum(supply*hiksian_demand)
            demand_agent = (epsilon)*(np.nan_to_num(weights[agent]/expenditure))*hiksian_demand
            allocation[agent,:] += ((demand+demand_agent) < supply)*demand_agent
            demand = np.sum(allocation, 0)
    return allocation


# %%
# Run Analysis of Different results
# Set parameters
## Experiment parameters
num_experiments = 20
## Market Parameters
num_goods = 5
num_agents = 3
rho = 0.5
epsilon = 0.001
supply = np.array([1, 1, 1, 1, 1])
weights = np.array([1/4, 1/2, 1/4])
valuations = np.random.rand(num_agents, num_goods)

# Test
X = run_drf(num_goods, num_agents, valuations, rho, supply, weights, epsilon = 1, prices = 1/supply)
print(np.sum(X, axis = 0))

# %%

# Compare to competitive equilibrium


ce_dominates = 0
drf_dominates = 0

ce_welfare = []
drf_welfare =[]
for i in range(500):
    if(not (i % 50)):
        print(f"--- Experiment {i} ---\n")
    # Create Market
    valuations = np.random.rand(num_agents, num_goods)
    market = m.FisherMarket(valuations, weights)
    
    # Solve for market prices and allocations for desired utility function structure.
    alloc_fm, p = market.solveMarket("leontief", printResults=False)
    alloc_drf = run_drf(num_goods, num_agents, valuations, rho, supply, weights, epsilon = 1, prices = 1/supply)    
    
    ce_dominates += pareto_dominates(alloc_fm, alloc_drf, valuations)
    drf_dominates += pareto_dominates(alloc_drf, alloc_fm, valuations)
    ce_welfare.append(get_leontief_welfare(alloc_fm, valuations))
    drf_welfare.append(get_leontief_welfare(alloc_drf, valuations))

print(f"Percentage of CE allocations dominating {ce_dominates/(i+1)}")
print(f"Percentage of drf allocations dominating {drf_dominates/(i+1)}")

plt.figure(figsize=(8,6))
plt.hist(ce_welfare, bins=50, alpha=0.5, label="Competitive Eqm. Welfare")
plt.hist(drf_welfare, bins=50, alpha=0.5, label="DRF Welfare")

plt.xlabel("Welfare", size=14)
plt.ylabel("Count", size=14)
plt.title("Comparing the Welfares of DRF and Competitive Equilibria for Leontief Markets")
plt.legend(loc='upper right')

plt.savefig('leontief.png')
#%%
# Check for IR being respected

num_not_IR = 0

for i in range(100):
    if (not (i % 50) ):
        print(f"--- Experiment {i} ---\n")
    
    valuation = np.random.rand(num_agents,num_goods)
    alloc = run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 1, prices = 1/supply)
    alloc_is_not_ir = is_not_ir(alloc, weights, valuation, rho)
    if (alloc_is_not_ir):
        print("There is an IR violation")
    num_not_IR += alloc_is_not_ir

print(f"The percentage of allocations not respecting IR is {num_not_IR/(i+1)}")

# %%
# Check for Incentive-Compatibility Violation

violates_ic = 0
for model in range(100):
    print(f"--- Model {model}---\n")
    valuation = np.random.rand(num_agents, num_goods)
    alloc1 = run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 1, prices = 1/supply)
    misreport = np.copy(valuation)
    for experiment in range(1000):
        if(not (experiment % 50)):
            print(f"--- Experiment {experiment}---\n")
        misreport[0,:] = np.random.rand(num_goods)
        alloc2 = run_drf(num_goods, num_agents, misreport, rho, supply, weights, epsilon = 1, prices = 1/supply)
        is_not_ic = (get_leontief_utility(alloc2[0,:], valuation[0,:]) > get_leontief_utility(alloc1[0,:], valuation[0,:]))
        if(is_not_ic):
            violates_ic += is_not_ic
            print(f"Fake Utility {get_leontief_utility(alloc2[0,:], valuation[0,:])}")                   
            print(f"Real Utility {get_leontief_utility(alloc1[0,:], valuation[0,:])}")
            break


        


print(f"The percentage of allocations not respecting IC is {violates_ic/(model+1)}")

# %%
# Check if there are any allocations not respecting envy-freeness
num_violates_ef = 0
total = 0
for i in range(50):
    total += 1
    if(not (i % 50)):
        print(f"--- Experiment {i} ---")
    valuation = np.random.rand(num_agents, num_goods)
    alloc = run_drf(num_goods, num_agents, valuation, rho, supply, weights, epsilon = 1, prices = 1/supply)
    for agent1 in range(num_agents):
        for agent2 in range(num_agents):
            if (agent1 != agent2):
                alloc1 = alloc[agent1,:]
                alloc2 = alloc[agent2,:]
                is_not_ef = (1/weights[agent2])*(get_leontief_utility(alloc2, valuation[agent1,:])) > (1/weights[agent1])*(get_leontief_utility(alloc1, valuation[agent1,:]))
                if(is_not_ef):
                    print(f"Other utility {(1/weights[agent2])*(get_leontief_utility(alloc2, valuation[agent1,:]))}")
                    print(f"Real Utility {(1/weights[agent1])*(get_leontief_utility(alloc1, valuation[agent1,:]))}")
                num_violates_ef += is_not_ef


print(f"The percentage of allocations not respecting EF is {num_violates_ef/total}")

# %%
# Create Market
supply = np.array([1, 1])
weights = np.array([1/2, 1/2])
valuations = np.array([[2, 1],[1, 1]])
market = m.FisherMarket(valuations, weights)

# Solve for market prices and allocations for desired utility function structure.
alloc_fm, p = market.solveMarket("leontief", printResults=False)
alloc_drf = run_drf(2, 2, valuations, 0, supply, weights, epsilon = 1, prices = 1/supply)    

print(f"Competitive equilibrium pareto-dominates: {pareto_dominates(alloc_fm, alloc_drf, valuations)}")
print(f"DRF pareto dominates:{pareto_dominates(alloc_drf, alloc_fm, valuations)}")
# %%
