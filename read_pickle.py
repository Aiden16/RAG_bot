import pickle

with open(f"C:/Users/njkro/OneDrive/Desktop/Learnings/PDFQA/vector_store/Resume2025/index.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(list(data.keys()) if isinstance(data, dict) else data)
print(data)
