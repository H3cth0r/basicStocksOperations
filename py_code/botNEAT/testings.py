import Trader as td


print("Hello!")

# Creating the trader bot
trader = td.Trader("NVDA", 100)

trader.prepareData()
"""
for i in trader.input_list:
    for j in i:
        print("{0:.3f}".format(j), end="\t\t")
    print()
"""
counter = 0
"""
while len(trader.input_list) > 0:
    print(f"len: {len(trader.input_list)}", end="\t\t")
    trader.input_list.pop()
    print(counter)
    counter += 1
"""

"""
while len(trader.input_list) > 0:
    print(f"len: {len(trader.input_list)}", end="\t\t")
    trader.data()
    print(counter)
    counter += 1
"""
#print(trader.input_list)
lista = trader.data()
print(f"len one row data = {len(lista)}")

print("End")
