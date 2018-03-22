import sqlite3
from os import mkdir
from os.path import exists, join
from pathlib import Path
import numpy as np
import model_assembly as ma
import pickle

def create_parameters_table():
    conn = sqlite3.connect("architectures.db", detect_types=sqlite3.PARSE_DECLTYPES)
    
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS parameters""")
    c.execute("""CREATE TABLE parameters (motif_id INT, input_shape TEXT, weights_file TEXT, biases_file TEXT, 
    FOREIGN KEY(motif_id) REFERENCES motifs(rowid))""")
    
    conn.commit()
    conn.close()
    
def create_connections_table():
    conn = sqlite3.connect("architectures.db", detect_types=sqlite3.PARSE_DECLTYPES)
    
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS connections""")
    c.execute("""CREATE TABLE connections (higher_level_motif_id INT, component_motif_id INT,
    FOREIGN KEY(higher_level_motif_id) REFERENCES motifs(higher_level_motif_id),
    FOREIGN KEY(component_motif_id) REFERENCES motifs(component_motif_id))""")
    
    conn.commit()
    conn.close()
    
def create_motifs_table():
    conn = sqlite3.connect("architectures.db", detect_types=sqlite3.PARSE_DECLTYPES)
    
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS motifs""")
    c.execute("""CREATE TABLE motifs (motif_filename TEXT, level INT, accuracy FLOAT, 
        full_weights_filename TEXT, ready INT, trained INT)""")
    
    conn.commit()
    conn.close()
    
def create_tables():
    create_motifs_table()
    create_parameters_table()
    create_connections_table()
    
def create_folders():
    motifs_path = "motifs"
    if not exists(motifs_path):
        mkdir(motifs_path)
        
    full_weights_path = join(motifs_path, "full_weights")
    if not exists(full_weights_path):
        mkdir(full_weights_path)
        
    params_path = "parameters"
    if not exists(params_path):
        mkdir(params_path)

def select(statement):
    conn = sqlite3.connect("architectures.db", detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(statement)
    results = c.fetchall()
    conn.commit()
    conn.close()
    return results

def insert(statement, insert):
    conn = sqlite3.connect("architectures.db", detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    #print("Inserting:", insert)
    c.execute(statement, insert)
    results = c.lastrowid
    conn.commit()
    conn.close()
    return results

def update(statement):
    conn = sqlite3.connect("architectures.db", detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    #print("Updating:", statement)
    c.execute(statement)
    conn.commit()
    conn.close()
    
def merge_weights(first_weights, second_weights):
    raise Exception("Look into better way of merging weights")
    params = np.divide(np.add(np.load(params_filename), params), 2)
    
def save_motif_weights(motif_id):
    full_weights_filename = join(full_weights_path, str(motif_id) + "~" + "fullweights.h5")
    model.save(full_weights_filename)
    update("""UPDATE motifs SET full_weights_filename = '{}' WHERE rowid = {}""".format(full_weights_filename, motif_id))
    
def save_motif(graph, motif_id, model):
    motif_filename = join("motifs", str(motif_id) + ".p")
    
    with open(motif_filename, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    
    return motif_filename

def insert_motif(graph, level, model = None, accuracy = 0.10):
    #motif_filename, level, accuracy, full_weights_filename, ready (0 = false), trained
    motif_insert = ('', level, accuracy, '', 0, 0)
    
    motif_id = insert("""INSERT INTO motifs VALUES (?, ?, ?, ?, ?, ?)""", motif_insert)
    
    motif_filename = save_motif(graph, motif_id, model)
    
    update("""UPDATE motifs SET motif_filename = '{}', ready = 1 WHERE rowid = {}""".format(motif_filename, motif_id))
    
    return motif_id

def update_motif(motif_id, accuracy, model = None):
    full_weights_filename = save_motif(graph, motif_id, model)
    
    update("""UPDATE motifs SET accuracy = {}, full_weights_filename = '{}' WHERE rowid = {}""".format(accuracy, full_weights_filename, motif_id))

def save_parameters(layer, name, input_shape_str, param_type, motif_id):
    if param_type is "weights":
        params = layer.get_weights()[0]
    else:
        params = layer.get_weights()[1]
    params_filename = join("parameters", str(motif_id) + "~" + name, param_type + "_" + input_shape_str + "-0.npy")
    path = Path(params_filename)
    
    i = 1
    while True:
        if path.is_file():
            params_filename = params_filename.split("-")[0] + "-%d" % (i) + ".npy"
            path = Path(params_filename)
        else:
            np.save(params_filename, params)
            break
        i += 1
    return params_filename

def insert_parameters(model):
    raise Exception("Need to finish")
    
    parameter_inserts = []
    for layer in model.layers:
        if layer.trainable:
            name = "".join(layer.name.split("_")[:-1])
            if name in primitives:
                if not exists(join("parameters", name)):
                    mkdir(join("parameters", name))
                input_shape_str = "_".join(map(str, layer.input_shape[1:]))

                weights_filename = save_parameters(layer, name, input_shape_str, "weights")
                biases_filename = save_parameters(layer, name, input_shape_str, "biases")

                parameter_inserts.appends((motif_id, 
                                          input_shape_str, 
                                          weights_filename, 
                                          biases_filename))  

        insert("""INSERT INTO parameters VALUES (?, ?, ?, ?)""", parameter_inserts)
        
def insert_set_of_working_motifs():
    motif_graphs = [
        [
            ['', 'conv2d3x3', ''],
            ['', '', 'dense']
        ],
                [
            ['', 'conv2d3x3', ''],
            ['', '', 'averagepooling2d3x3']
        ],
        [
            ['', 'separableconv2d3x3', ''],
            ['', '', 'averagepooling2d3x3']
        ]
        ,[
            ['', 'dense', ''],
            ['', '', 'dense']
        ]
    ]
    
    for G in motif_graphs:
        insert_motif(G, ma.assemble_model(G), level = 1)