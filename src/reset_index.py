"""
One-off: clear every namespace in the Pinecone index (experiment cfg_* + default),
leaving the index itself intact and ready for a clean full-corpus build.

Run this ONCE, then run indexer.py.
"""

from vector_store import index

before = index.describe_index_stats()
namespaces = list(before.get("namespaces", {}).keys())

if not namespaces:
    print("Index already empty — nothing to clear.")
else:
    for ns in namespaces:
        index.delete(delete_all=True, namespace=ns)
        print("cleared:", ns or "(default)")

print("\nAfter reset:")
print(index.describe_index_stats())
