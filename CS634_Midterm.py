import pandas as pd
import itertools
import time
import os
from mlxtend.frequent_patterns import apriori, fpgrowth

#print("Current working directory:", os.getcwd())


def load_data(dataset_name):
    if dataset_name == 'Amazon':
        transactions = pd.read_csv('amazon_transactions.csv', encoding = 'ISO-8859-1')
        items = pd.read_csv('amazon_items.csv', encoding = 'ISO-8859-1')
    elif dataset_name == 'Best Buy':
        transactions = pd.read_csv('bestbuy_transactions.csv', encoding = 'ISO-8859-1')
        items = pd.read_csv('bestbuy_items.csv', encoding = 'ISO-8859-1')
    elif dataset_name == 'Kmart':
        transactions = pd.read_csv('kmart_transactions.csv', encoding = 'ISO-8859-1')
        items = pd.read_csv('kmart_items.csv', encoding = 'ISO-8859-1')
    elif dataset_name == 'Nike':
        transactions = pd.read_csv('nike_transactions.csv', encoding = 'ISO-8859-1')
        items = pd.read_csv('nike_items.csv', encoding = 'ISO-8859-1')
    elif dataset_name == 'General':
        transactions = pd.read_csv('general_transactions.csv', encoding = 'ISO-8859-1')
        items = pd.read_csv('general_items.csv', encoding = 'ISO-8859-1')
    else:
        raise ValueError("Invalid dataset name")
    return transactions, items

def preprocess_transactions(transactions):
    unique_items = sorted(set(itertools.chain.from_iterable(transactions['items'].apply(lambda x: x.split(',')))))
    return unique_items

def get_min_support_count(transactions, min_support):
    return int(min_support * len(transactions))

def generate_candidates(itemsets, length):
    return set(itertools.combinations(itemsets, length))

def count_itemsets(transactions, candidates):
    itemset_count = {itemset: 0 for itemset in candidates}
    for transaction in transactions['items']:
        transaction_items = set(transaction.split(','))
        for itemset in candidates:
            if set(itemset).issubset(transaction_items):
                itemset_count[itemset] += 1
    return {itemset: count for itemset, count in itemset_count.items() if count > 0}

def generate_frequent_itemsets(transactions, min_support_count):
    itemsets = [tuple([item]) for item in preprocess_transactions(transactions)]
    frequent_itemsets = {}
    k = 1
    max_k = len(set(item for transaction in transactions for item in transaction))  # Maximum itemset size

    for k in range(1, max_k + 1):
        candidates = generate_candidates(itemsets, k)
        itemset_counts = count_itemsets(transactions, candidates)

        # Update frequent itemsets based on support count
        frequent_itemsets.update({
            itemset: count 
            for itemset, count in itemset_counts.items() 
            if count >= min_support_count
        })
        
        itemsets = list(itemset_counts.keys())
        
        # If there are no frequent itemsets left, we can break early
        if not itemsets:
            break

    return frequent_itemsets
    

def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset, support in frequent_itemsets.items():
        for i in range(1, len(itemset)):
            for subset in itertools.combinations(itemset, i):
                antecedent = subset
                consequent = tuple(set(itemset) - set(antecedent))
                if consequent:
                    confidence = support / frequent_itemsets[antecedent]
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    return rules
     

def main():
    dataset_name = input("Select a dataset (Amazon, Best Buy, Kmart, Nike, General): ")
    transactions, items = load_data(dataset_name)
    
    min_support = float(input("Enter minimum support count (as a percentage): ")) / 100
    min_confidence = float(input("Enter minimum confidence (as a decimal): "))
    
    min_support_count = get_min_support_count(transactions, min_support)
    
    start_time = time.time()
    frequent_itemsets = generate_frequent_itemsets(transactions, min_support_count)
    rules = generate_association_rules(frequent_itemsets, min_confidence)
    end_time = time.time()
    
    print("Frequent Itemsets:")
    for itemset, support in frequent_itemsets.items():
        print(f"{itemset}: {support}")
    
    print("\nAssociation Rules:")
    for antecedent, consequent, confidence in rules:
        print(f"{antecedent} -> {consequent} (Confidence: {confidence})")
    
    print(f"\nExecution Time: {end_time - start_time} seconds")
    
    # Alternative Methods using mlxtend
    transactions_encoded = pd.get_dummies(transactions['items'].str.get_dummies(sep=','))
    frequent_itemsets_apriori = apriori(transactions_encoded, min_support=min_support, use_colnames=True)
    frequent_itemsets_fpgrowth = fpgrowth(transactions_encoded, min_support=min_support, use_colnames=True)

    print("\nFrequent Itemsets using Apriori:")
    print(frequent_itemsets_apriori)
    
    print("\nFrequent Itemsets using FPGrowth:")
    print(frequent_itemsets_fpgrowth)

if __name__ == "__main__":
    main()
