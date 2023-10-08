import pickle


with open('rf.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)