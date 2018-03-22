import pickle
from tqdm import tqdm
from copy import copy
from IPython.core.debugger import set_trace

import model_assembly as ma
import database as db
from dataset import Dataset
dataset = Dataset()

percentage = 1.

train_generator, validation_generator, _, _ = dataset.create_generators(percentage)
train_amount = len(dataset.X_train[:int(len(dataset.X_train) * percentage)])

val_amount = len(dataset.X_val[:int(len(dataset.X_val) * percentage)])

usablility_threshold = .9

def train_motif(motif_id, motif, level, batch_size = 32, baseline_accuracy = .10):
    val_steps = val_amount // batch_size
    train_steps = train_amount // batch_size
    db.update("""UPDATE motifs SET trained = 1, accuracy = {}""".format(baseline_accuracy))
    
    current_acc = copy(baseline_accuracy)
    last_acc = 0.0
    model = ma.assemble_model(motif)

    i = 0
    while True:
        history = model.fit_generator(
        train_generator,
        steps_per_epoch = train_steps,
        epochs = 1,
        validation_data = validation_generator,
        validation_steps = val_steps,
        verbose = 2)

        current_acc = history.history['val_acc'][0]
        
        # idea: save internal weights of motifs, if possible initialize those to lower motifs.
        # connection to inputs and the outputs could be on a per primitive basis and be pulled from a common
        # set of weights
        
        if current_acc > save_threshold and current_acc > last_acc:
            print("Updating motif")
            if current_acc > usability_threshold:
                print("Saving full weights")
                save_motif_weights(motif_id)
            db.update_motif(motif_id, accuracy)

        if current_acc < baseline_accuracy * 1.05:
            break
        elif current_acc < last_acc * 1.01:
            break
        elif i > 50:
            break

        last_acc = copy(current_acc)

        i += 1 
        print("Training another epoch...")

def train_level(level):
    if level == 0:
        raise Exception("Cannot train primitives")
    
    motifs = [(motif[0], pickle.load( open(motif[1], 'rb'))) for motif in db.select("""SELECT rowid, motif_filename FROM motifs WHERE level={} and trained=0""".format(level))]
    
    print("Number of motifs:", len(motifs))
    
    for motif_id, motif in tqdm(motifs):
        train_motif(motif_id, motif, level)