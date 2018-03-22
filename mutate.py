import numpy as np
from database import select, insert_motif
from copy import copy, deepcopy
from dataset import Dataset
from model_assembly import assemble_model, get_primitives
from os.path import exists, join
from IPython.core.debugger import set_trace
import pickle
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

primitives = get_primitives()
dataset = Dataset()
train_generator, validation_generator, _, _ = dataset.create_generators(.1)

def successfully_builds(G):
    builds = False
    model = None
    try:
        model = assemble_model(G)
        builds = True
    except (KeyboardInterrupt, SystemExit):
        raise
    #except Exception as e:
    #    print(e, G)
    
    return builds, model

def successfully_runs(model):
    runs = False
    global train_generator, validation_generator
    try:
        train_steps = 1
        val_steps = 1
        model.fit_generator(
            train_generator,
            steps_per_epoch = train_steps,
            epochs = 1,
            validation_data = validation_generator,
            validation_steps = val_steps,
            verbose = 2)
        runs = True
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(e)
        model.summary()
    
    return runs

def check_viability(motif, same_level_motifs):
    if motif in same_level_motifs:
        return False, _
    builds, model = successfully_builds(motif)
    if not builds:
        return False, _
    #if not successfully_runs(model):
    #    return False, _
        
    return True, model

def mutate(motif, lower_level_motifs, same_level_motifs):
    random_predecessor_i = np.random.randint(0, len(motif) - 1)
    random_successor_j = np.random.randint(random_predecessor_i + 1, len(motif[1]) - 1)

    mutation = str(lower_level_motifs[np.random.choice(len(lower_level_motifs))])

    motif[random_predecessor_i][random_successor_j] = mutation
    
    works, _ = check_viability(motif, same_level_motifs)
    
    return works
    
def mutate_motifs(level, num_mutants, num_threads):
    if level == 0:
        raise Exception("Cannot mutate primitives")
    elif level == 1:
        lower_level_motifs = copy(primitives)
    else:
        lower_level_motifs = [(lower_motif[0], pickle.load( open(lower_motif[1], 'rb'))) for lower_motif in select("""SELECT rowid FROM motifs WHERE level={}""".format(level - 1))]

    same_level_motifs = [pickle.load( open(same_level_motif[1], 'rb')) for same_level_motif in select("""SELECT rowid, motif_filename FROM motifs WHERE level={}""".format(level))]
        
    num_inserted = 0
    def task(_):
        nonlocal num_inserted
        if num_inserted < num_mutants:
            motif = same_level_motifs[np.random.choice(len(same_level_motifs))]
            works = mutate(motif, lower_level_motifs, deepcopy(same_level_motifs))
            if works:
                insert_motif(motif, level)
                num_inserted += 1
                
    while True:
        if num_threads > 1:
            with ThreadPool(num_threads) as pool:
                pool.map(task, range(num_mutants - num_inserted))
        else:
            for _ in tqdm(range(num_mutants - num_inserted)):
                task(_)
        if num_inserted >= num_mutants:
            print("Finished mutating motifs")
            return
        same_level_motifs = [pickle.load( open(same_level_motif[1], 'rb')) for same_level_motif in select("""SELECT rowid, motif_filename FROM motifs WHERE level={}""".format(level))]
            
def create_motif(level, lower_level_motifs, same_level_motifs, graph_shape = (2, 3)):
    motif = [
        ['', '', ''], 
        ['', '', '']
    ]
    for i in range(graph_shape[0]):
        for j in range(graph_shape[1]):
            if j > i:
                mutation = str(np.random.choice(lower_level_motifs))

                motif[i][j] = mutation

                works, _ = check_viability(motif, same_level_motifs)
    return works, motif
    
def create_random_motifs(level, num_mutants, num_threads):
    if level == 0:
        raise Exception("Cannot create random pool from primitives")
    elif level == 1:
        lower_level_motifs = copy(primitives)
    else:
        lower_level_motifs = [motif[0] for motif in select("""SELECT rowid FROM motifs WHERE level={}""".format(level - 1))]
        if len(lower_level_motifs) < 1:
            raise Exception("No lower level motifs exist, please call create_working_set_of_mutants for the previous level.")

    same_level_motifs = [pickle.load( open(same_level_motif[1], 'rb')) for same_level_motif in select("""SELECT rowid, motif_filename FROM motifs WHERE level={}""".format(level))]
    num_inserted = 0
    
    def task(_):
        nonlocal num_inserted
        if num_inserted < num_mutants:
            works, motif = create_motif(level, lower_level_motifs, same_level_motifs)
            if works:
                insert_motif(motif, level)
                num_inserted += 1
    
    while True:
        if num_threads > 1:
            with ThreadPool(num_threads) as pool:
                pool.map(task, range(num_mutants - num_inserted))
        else:
            for _ in tqdm(range(num_mutants - num_inserted)):
                task(_)
        if num_inserted >= num_mutants:
            print("Finished inserting random motifs")
            return
        same_level_motifs = [pickle.load( open(same_level_motif[1], 'rb')) for same_level_motif in select("""SELECT rowid, motif_filename FROM motifs WHERE level={}""".format(level))]
                
def create_random_mutants(level, num_mutants = 100, mutate_existing = False, num_threads = 3):
    motifs = select("""SELECT COUNT(*) FROM motifs""")
    if len(motifs) < 1:
        if mutate_existing:
            raise Exception("Need at least one motif to mutate existing")
        else:
            create_random_motifs(level, num_mutants, num_threads = num_threads)
    else:
        if mutate_existing:
            mutate_motifs(level, num_mutants = num_mutants, num_threads = num_threads)
        else:
            create_random_motifs(level, num_mutants, num_threads = num_threads)
    else:
        print("Level {} mutants ready.".format(level))
        return True