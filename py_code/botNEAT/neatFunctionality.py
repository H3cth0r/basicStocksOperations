
import neat
import sys
import os

from Trader import Trader

def remove(index):
    traders.pop(index)
    ge.pop(index)
    nets.pop(index)

def evaluate_genomes(genomes, config):
    global traders, ge, nets 

    traders     =   []
    ge          =   []
    nets        =   []

    for genome_id, genome in genomes:
        traders.append(Trader())                    # TODO check initialization of Trader
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0
    
    run = True
    while run:
        
        if len(traders) == 0:
            break

        for i, trader in enumerate(traders):
            ge[i].fitness += 1
            if not trader.alive:
                remove(i)
        
        # Buy or sell
        for i, trader in enumerate(traders):
            output = nets[i].activate(trader.data())    # TODO data method not working
            if output[0] > 0.7:
                trader.buy(output[2])                   # TODO create output[2] of config file
            if output[1] > 0.7:
                trader.sell(output[2])
            if output[0] <= 0.7 and output[1] <= 0.7:
                continue
                # trader.buy_sell = 0
        
        # Update
        for trader in traders:
            trader.update()

def run(config_path):
    pass