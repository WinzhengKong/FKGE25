from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from server import Server
from client import Client
import os
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

class Controller:
    def __init__(self):
        return
    
    def parse_args(self,args=None):
        parser = argparse.ArgumentParser(
            description="train, valid, test and unlearn kge models collaboratively",
            usage = "controller_byzantine.py [<args>] [-h | --help]"
        )
        parser.add_argument("--client_num",type=int,default=3,help="client num, type int, default 3")
        parser.add_argument("--local_file_dir",type=str,default="../Data/FB15k-237/R3FL",help="local file directory, type str, default ../Data/FB15k-237/C3FL")
        parser.add_argument("--save_dir",type=str,default="../Output/FB15k-237/R3FL",help="save dir, type str, default ../Output/FB15k-237/C3FL")
        parser.add_argument("--aggregate_iteration",type=int,default=200,help="aggregate iterations, type int, default 200")
        #parser.add_argument("--cuda",action="store_true",help="use GPU, store true")
        parser.add_argument("--cuda", default=True, action="store_false", help="use GPU if available, default is True")
        parser.add_argument("--model",type=str,default="TransE",help="model, type str, choose between TransE/DisMult/ComplEx/RotatE")
        parser.add_argument("--double_entity_embedding",action="store_true",help="double entity embedding, store true")
        parser.add_argument("--double_relation_embedding",action="store_true",help="double relation embedding, store true")
        parser.add_argument("--max_epoch",type=int,default=400,help="max epoch, type int, default 400")
        parser.add_argument("--valid_epoch",type=int,default=10,help="valid epoch, type int, default 10")
        parser.add_argument("--early_stop_epoch",type=int,default=15,help="early stop epoch, type int, default 15")
        parser.add_argument("--cpu_num",type=int,default=16,help="cpu num, type int, default 16")
        parser.add_argument("--negative_sample_size",type=int,default=256,help="negative sample size, type int, default 256")
        parser.add_argument("--negative_adversarial_sampling",action="store_true",help="negative adversarial sampling, store true")
        parser.add_argument("--adversarial_temperature",type=float,default=1.0,help="float, adversarial temperature, default 1.0")
        parser.add_argument("--uni_weight",action="store_true",help="uni weight, store true")
        parser.add_argument("--regularization",type=float,default=0.0,help="regularization, type float, default 0.0")
        parser.add_argument("--batch_size",type=int,default=1024,help="batch size, type int, default 1024")
        parser.add_argument("--hidden_dim",type=int,default=256,help="hidden dim, type int, default 256")
        parser.add_argument("--learning_rate",type=float,default=1e-4,help="learning rate, type float, default 1e-4")
        parser.add_argument("--gamma",type=float,default=9.0,help="gamma, type float, default 9.0")
        parser.add_argument("--epsilon",type=float,default=2.0,help="epsilon, type float, default 2.0")
        parser.add_argument("--test_batch_size",type=int,default=64,help="test batch size, type int, default 32")
        parser.add_argument("--log_epoch",type=int,default=10,help="log epoch, type int, default 10")
        parser.add_argument("--test_log_step",type=int,default=200,help="test log step, type int, default 200")
        parser.add_argument("--fed_mode",type=str,default="FedAvg",help="fed mode, type str, choose from FedAvg/FedProx/FedDist")
        parser.add_argument("--mu",type=float,default=0.0,help="mu, type float, default 0.0")
        parser.add_argument("--mu_decay",action="store_true",help="mu decay, store true")
        parser.add_argument("--mu_single_entity",action="store_true",help="mu single entity, store true")
        parser.add_argument("--eta",type=float,default=1.0,help="eta, type float, default 1.0")
        parser.add_argument("--agg",type=str,default="weighted",help="aggregation method, type str, default weighted, optional weighted/distance/similarity")
        parser.add_argument("--max_iter",type=int,default=300,help="max iter, type int, default 300")
        parser.add_argument("--valid_iter",type=int,default=5,help="valid iter, type int, default 5")
        parser.add_argument("--early_stop_iter",type=int,default=15,help="early stop iter, type int, default 15")
        parser.add_argument("--dist_mu",type=float,default=1e-2,help="distillation mu, type float, default 1e-2")
        parser.add_argument("--co_dist",action="store_true",help="co-distillation, store true")
        parser.add_argument("--wait_iter",type=int,default=10)
        parser.add_argument("--mu_contrastive",type=float,default=0.1)
        parser.add_argument("--mu_temperature",type=float,default=0.2)
        parser.add_argument("--byzantine", type=str, default="RandomNoise",
                            help="byzantine attack, type str, choose from RandomNoise/AddNoise/VectorFlip/PartialFlip/Poison")
        parser.add_argument("--adm", type=str, default="ECOD",
                            help="Anomaly Detection Model, type str, choose from IsolationForest(IF)/ECOD/None")
        parser.add_argument("--component1", type=str, default="normalization",
                            help="normalization, type str, choose from normalization/None")
        args = parser.parse_args(args)
        if args.local_file_dir is None:
            raise ValueError("local file dir must be set")
        if args.fed_mode=="FedDist":
            args.eta=0.0
        elif args.fed_mode in ["FedAvg","FedProx","FedEC"]:
            args.eta=1.0
        if args.model=="RotatE":
            args.double_entity_embedding=True
            args.negative_adversarial_sampling=True
        elif args.model=="ComplEx":
            args.double_entity_embedding=True
            args.double_relation_embedding=True
        self.args = args
    
    def init_federation(self):
        args = self.args
        server = Server(args)
        clients = []
        for i in range(args.client_num):
            client = Client(i,args)
            clients.append(client)
        self.server = server
        self.clients = clients
    
    def init_model(self):
        args = self.args
        self.server.generate_global_embedding()
        client_embedding_dict = self.server.assign_embedding()
        for i in range(args.client_num):
            self.clients[i].init_model(init_entity_embedding=client_embedding_dict[i])
        
    def train(self):
        args = self.args
        for i in range(args.aggregate_iteration):
            client_embedding_dict = dict()
            metrics = []
            for i in range(args.client_num):
                metrics.append(self.clients[i].train_round())
                client_embedding_dict[i] = self.clients[i].get_entity_embedding()
            log_metrics("Test After Local Training",0,metrics)
            if i!=args.aggregate_iteration-1:
                print('Fed aggregate!!!')
                self.server.aggregate_embedding(client_embedding_dict)
                client_embedding_dict = self.server.assign_embedding()
                for client_seq in client_embedding_dict.keys():
                    self.clients[client_seq].update_model(entity_embedding=client_embedding_dict[client_seq])
        
    def save(self):
        args = self.args
        for i in range(0,args.client_num):
            self.clients[i].save_model({})
    def load(self):
        args = self.args
        for i in range(0,args.client_num):
            self.clients[i].load_model()
'''                
def log_metrics(mode,iter,logs):
    print("-"*20+"\n")
    print("%s in Iter %i"%(mode,iter))
    for i in range(0,len(logs)):
        print("Log of Client %i"%(i))
        for metric in logs[i].keys():
            if metric!="n":
                print("%s:%f"%(metric,logs[i][metric]))
    print("Weight Averaged of All Clients")
    for metric in logs[0].keys():
        if metric!="n":
            weighted_metric = sum([log[metric]*log["n"] for log in logs])/sum([log["n"] for log in logs])
            print("%s:%f"%(metric,weighted_metric))
'''
def log_metrics(mode, iter, logs, byzantine_client_indices=[]):
    print("-" * 20 + "\n")
    print("%s in Iter %i" % (mode, iter))

    # Print logs for each client
    for i in range(0, len(logs)):
        '''
        if i in byzantine_client_indices:
            continue  # Skip Byzantine clients
        '''
        print("Log of Client %i" % (i))
        for metric in logs[i].keys():
            if metric != "n":  # Skip the 'n' metric
                print("%s:%f" % (metric, logs[i][metric]))

    print("Weight Averaged of All Clients")

    # Calculate weighted metrics, excluding Byzantine clients
    for metric in logs[0].keys():
        if metric != "n":  # Skip the 'n' metric
            weighted_metric = sum(
                [log[metric] * log["n"] for i, log in enumerate(logs) if i not in byzantine_client_indices]) / \
                              sum([log["n"] for i, log in enumerate(logs) if i not in byzantine_client_indices])
            print("%s:%f" % (metric, weighted_metric))

if __name__=="__main__":
    controller = Controller()
    controller.parse_args()
    controller.init_federation()
    controller.init_model()
    controller.train()