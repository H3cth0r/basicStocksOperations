
import neat
import os

import matplotlib.pyplot as plt


import sys
sys.path.append("..")
import stocksOps as so

from Trader import Trader


# Variable for storing best credit score 
best_credit_score = 100




def remove(index):
    traders.pop(index)
    ge.pop(index)
    nets.pop(index)

def evaluate_genomes(genomes, config):
    global traders, ge, nets 

    traders     =   []
    ge          =   []
    nets        =   []

    df = so.downloadIntradayData("NVDA")

    for genome_id, genome in genomes:
        traders.append(Trader("NVDA", 100, df.copy(deep=True), genome_id))                    # FIXME fix initialization
        traders[-1].prepareData()                        # Data preparartion

        ge.append(genome)

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

        genome.fitness = 0
    
    run = True
    while run:
        #print(f"Counter ======================= {counter}")
        if len(traders) == 0:
            break
        
        """
        The alive condition must be True while or credit is more than 0 
        """
        for i, trader in enumerate(traders):
            """
            if trader.credit > trader.original_credit:
                ge[i].fitness += 1
            """
            if trader.credit > trader.last_credit * 1.09:
                ge[i].fitness += 5
            elif trader.credit > trader.last_credit * 1.07:
                ge[i].fitness += 4
            elif trader.credit > trader.last_credit * 1.05:
                ge[i].fitness += 3
            elif trader.credit > trader.last_credit * 1.03:
                ge[i].fitness += 2
            elif trader.credit > trader.last_credit:
                ge[i].fitness += 1
            
            if trader.credit + trader.getEarnings() > trader.original_credit:
                ge[i].fitness += 3

            # If the credit is greater than the history of best credit score
            global best_credit_score
            #if trader.credit > best_credit_score * 0.97 and best_credit_score * 0.97 > trader.original_credit:
            if trader.credit > best_credit_score:
                ge[i].fitness += 5
            elif trader.credit > best_credit_score * 0.97 and best_credit_score*0.97 > trader.original_credit: 
                ge[i].fitness += 4
            elif trader.credit > best_credit_score * 0.95 and best_credit_score*0.95 > trader.original_credit: 
                ge[i].fitness += 3
            elif trader.credit > best_credit_score * 0.93 and best_credit_score*0.93 > trader.original_credit: 
                ge[i].fitness += 2
            elif trader.credit > best_credit_score * 0.91 and best_credit_score*0.91 > trader.original_credit: 
                ge[i].fitness += 1

            if not trader.alive():
                remove(i)
        
        # Buy or sell
        """
        The output of the neural network must represent the
        quantity to buy or sell.
        If both buy and sell outputs are equal to zero, then 
        do nothing.

        The output is multiplied by 1000, so the for example
        0.142 * 1000 = 142 qty
        """
        for i, trader in enumerate(traders):
            if len(trader.input_list) == 0:
                td = traders[0]
                for i in traders:
                    if i.credit > td.credit:
                        td = i
                print(f"-> This is the credit of \"{td.id}\" trader : {td.credit}")
                
                # updating best output credit score if apply
                if td.credit > best_credit_score:best_credit_score=td.credit
                #if td.credit > best_credit_score or best_credit_score * 0.93 > td.credit:best_credit_score=td.credit

                """
                plt.figure(figsize=(25,8))
                plt.plot(td.credit_hist, '-gD', label = "credit")

                #plt.show()
                #plt.plot(td.holdings_hist)
                plt.plot(td.selled_hist, label = "selled")
                plt.plot(td.bought_hist, label = "bought")
                plt.legend(loc='upper center')
                plt.show()
                """
                return
            output = nets[i].activate(trader.data())

            if output[0] > 0:
                trader.buy(output[0])
            if output[1] > 0:
                trader.sell(output[1])
            #print(f"output 0 : {output[0]}\t\toutput 1 : {output[1]}")

            # Pop row on on closings
            trader.closings.pop(0)

        # Update
        # FIXME might not need this method here, might just apply the update on the 
        # Buy and sell methods.
        for trader in traders:
            trader.update()
    

def run(config_path):
    global pop
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)


    pop.run(evaluate_genomes, 50)