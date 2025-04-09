import torch.nn as nn
from adasyn import AdaSyn
from sgru import SGRU
from adamem import AdaMem
import snntorch as snn
import snntorch.spikegen as sgen
from syn import CustomSynaptic

import torch

class ADAMEMNET(nn.Module):
    def __init__(self,mode="FULL"):
        super().__init__()
        if mode == "FULL":
            self.fc1 = nn.Linear(100*100,3)
            self.timesteps = 200
        elif mode == "ROWS_COLUMNS":
            self.fc1 = nn.Linear(200,3)
        else:
            self.fc1 = nn.Linear(100,3)
        
        self.lif1 = AdaMem(0.99,1.0,5.0,0.95,learn_beta=False,learn_threshold_decay=True,learn_scale=False,base_threshold=0.0)
        self.mode = mode

    def forward(self,x):
        mem,thr = self.lif1.reset_mem()
        
        
        spk_rec, mem_rec, syn_rec, thr_rec = [],[],[],[]

        if self.mode == "ROWS_COLUMNS":
            rate_coded = sgen.rate(x,1)[0]
            for index in range(x.shape[1]):
                row = rate_coded[:,index,:]
                col = rate_coded[:,:,index]
                row_col = torch.concatenate((row,col),dim=1)
                fc1 = self.fc1(row_col)
                spk,mem,thr = self.lif1(fc1,mem,thr)
                
                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "FULL":
            rate_coded_input = sgen.rate(x.flatten(1),self.timesteps)
        
            for t in range(self.timesteps):            
                fc1_out = self.fc1(rate_coded_input[t])
                spk, mem,thr  = self.lif1(fc1_out, mem,thr )

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_ROWS":
            rate_coded = sgen.rate(x,1)[0]
            for col in range(x.shape[1]):
            
                fc1_out = self.fc1(rate_coded[:,:,col])
                spk, mem,thr  = self.lif1(fc1_out,mem,thr )

            
        
                spk_rec.append(spk)
                mem_rec.append(mem)
        else:
            rate_coded = sgen.rate(x,1)[0]
            for row in range(x.shape[1]):
            
                fc1_out = self.fc1(rate_coded[:,row,:])
                spk, mem,thr   = self.lif1(fc1_out,mem,thr )

            
        
                spk_rec.append(spk)
                mem_rec.append(mem)
        return torch.stack(spk_rec), torch.stack(mem_rec)

class ADASYNNET(nn.Module):
    def __init__(self,mode="FULL"):
        super().__init__()
        if mode == "FULL":
            self.fc1 = nn.Linear(100*100,3)
            self.timesteps = 200
        elif mode == "ROWS_COLUMNS":
            self.fc1 = nn.Linear(200,3)
        else:
            self.fc1 = nn.Linear(100,3)
        
        self.adaptive1 = AdaSyn(alpha=0.9,beta=0.8,beta_scale=1.0,threshold=1.0,threshold_decay=0.9,learn_alpha=True,learn_beta=True,learn_threshold_decay=True,base_threshold=1.0)
        self.mode = mode

    def forward(self,x):
        syn,mem,thr = self.adaptive1.reset_mem()
        
        
        spk_rec, mem_rec, syn_rec, thr_rec = [],[],[],[]
        
        
        if self.mode == "ROWS_COLUMNS":
            rate_coded = sgen.rate(x,1)[0]
            for index in range(x.shape[1]):
                row = rate_coded[:,index,:]
                col = rate_coded[:,:,index]
                row_col = torch.concatenate((row,col),dim=1)
                fc1_out = self.fc1(row_col)
                spk, mem,syn,thr  = self.adaptive1(fc1_out,mem,syn, thr)

            
        
                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "FULL":
            rate_coded_input = sgen.rate(x.flatten(1),self.timesteps)
        
            for t in range(self.timesteps):            
                fc1_out = self.fc1(rate_coded_input[t])
                spk, mem,syn,thr = self.adaptive1(fc1_out, mem,syn, thr)

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_ROWS":
            rate_coded = sgen.rate(x,1)[0]
            for col in range(x.shape[1]):
            
                fc1_out = self.fc1(rate_coded[:,:,col])
                spk, mem,syn,thr  = self.adaptive1(fc1_out,mem,syn, thr)

            
        
                spk_rec.append(spk)
                mem_rec.append(mem)
        else:
            rate_coded = sgen.rate(x,1)[0]
            for row in range(x.shape[1]):
            
                fc1_out = self.fc1(rate_coded[:,row,:])
                spk, mem,syn,thr  = self.adaptive1(fc1_out,mem,syn, thr)

            
        
                spk_rec.append(spk)
                mem_rec.append(mem)
        return torch.stack(spk_rec), torch.stack(mem_rec)

class SLSTMNET(nn.Module):
    def __init__(self,mode="FULL"):
        super().__init__()
        
        if mode == "ROWS_COLUMNS":
            self.slstm1 = snn.SLSTM(200,50,learn_threshold=True)
        elif mode == "3D":
            self.slstm1 = snn.SLSTM(100*100,50,learn_threshold=True)
        else:
            self.slstm1 = snn.SLSTM(100,50,learn_threshold=True)
        self.fc1 = nn.Linear(50,3)
        self.lif1 = snn.Leaky(beta=0.95,threshold=1.0,learn_beta=True, learn_threshold=True)
        self.timesteps = 100
        self.mode = mode
    def forward(self,x):
        syn, mem1 = self.slstm1.init_slstm()

        mem = self.lif1.init_leaky()
        spk_rec = []
        mem_rec = []

        if self.mode == "ROWS_COLUMNS":
            
            rate_coded = sgen.rate(x,1)[0]
            for index in range(x.shape[1]):
                row = rate_coded[:,index,:]
                col = rate_coded[:,:,index]
                row_col = torch.concatenate((row,col),dim=1)
                spk2, syn, mem1= self.slstm1(row_col, syn, mem1)

                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_ROWS":
            rate_coded = sgen.rate(x,1)[0]
            for col in range(x.shape[1]):
           
                spk2, syn, mem1= self.slstm1(rate_coded[:,:,col], syn, mem1)

                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_COLUMNS":
            rate_coded = sgen.rate(x,1)[0]
            for row in range(x.shape[1]):
           
                spk2, syn, mem1= self.slstm1(rate_coded[:,row,:], syn, mem1)

                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "3D":
            for timestep in range(x.shape[-1]):
                input_data = x[:,:,:,timestep].flatten(start_dim=1)
                spk2, syn, mem1= self.slstm1(input_data, syn, mem1)
                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
            

        return torch.stack(spk_rec), torch.stack(mem_rec)

class SGRUNET(nn.Module):
    def __init__(self,mode="FULL"):
        super().__init__()
        
        if mode == "ROWS_COLUMNS":
            self.sgru1 = SGRU(200,50, learn_threshold=True)
        elif mode == "3D":
            self.sgru1 = SGRU(100*100,50, learn_threshold=True)
        else:
            self.sgru1 = SGRU(100,50,learn_threshold=True)
        self.fc1 = nn.Linear(50,3)
        self.lif1 = snn.Leaky(beta=0.95,threshold=1.0,learn_beta=True, learn_threshold=True)
        self.timesteps = 200
        self.mode = mode
    def forward(self,x):
        sgrumem = self.sgru1.reset_mem()
        mem = self.lif1.init_leaky()
        spk_rec = []
        mem_rec = []
        
        if self.mode == "ROWS_COLUMNS":
            rate_coded = sgen.rate(x,1)[0]
            for index in range(x.shape[1]):
                row = rate_coded[:,index,:]
                col = rate_coded[:,:,index]
                row_col = torch.concatenate((row,col),dim=1)
                spk2,sgrumem = self.sgru1(row_col, sgrumem)

                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_ROWS":
            rate_coded = sgen.rate(x,1)[0]
            for col in range(x.shape[1]):
           
                spk2,sgrumem = self.sgru1(rate_coded[:,:,col], sgrumem)

                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "3D":
            for timestep in range(x.shape[-1]):
                input_data = x[:,:,:,timestep].flatten(start_dim=1)
                spk2,sgrumem = self.sgru1(input_data, sgrumem)
                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_COLUMNS":
            rate_coded = sgen.rate(x,1)[0]
            for row in range(x.shape[1]):
           
                spk2,sgrumem = self.sgru1(rate_coded[:,row,:], sgrumem)

                fc2_out = self.fc1(spk2)
                spk,mem = self.lif1(fc2_out)

                spk_rec.append(spk)
                mem_rec.append(mem)

        return torch.stack(spk_rec), torch.stack(mem_rec)


class LEAKYNETWORK(nn.Module):
    def __init__(self,mode="FULL"):
        super().__init__()
        if mode == "3D":
            self.fc1 = nn.Linear(100*100,3)
            self.timesteps = 200
        elif mode == "ROWS_COLUMNS":
            self.fc1 = nn.Linear(200,3)
        else:
            self.fc1 = nn.Linear(100,3)

        self.lif1 = snn.Leaky(0.95,1.0,learn_beta=True,learn_threshold=True)
        self.mode = mode

    def forward(self,x):
        mem = self.lif1.init_leaky()
        spk_rec = []
        mem_rec = []

        
        if self.mode == "ROWS_COLUMNS":
            rate_coded = sgen.rate(x,1)[0]
            for index in range(x.shape[1]):
                row = rate_coded[:,index,:]
                col = rate_coded[:,:,index]
                row_col = torch.concatenate((row,col),dim=1)
                fc1 = self.fc1(row_col)
                spk,  mem = self.lif1(fc1,mem)
                
                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "3D":
            for timestep in range(x.shape[-1]):
                input_data = x[:,:,:,timestep].flatten(start_dim=1)
                fc1 = self.fc1(input_data)
                spk,  mem = self.lif1(fc1,mem)
                
                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_ROWS":
            rate_coded = sgen.rate(x,1)[0]
            for col in range(x.shape[1]):
            
                fc1 = self.fc1(rate_coded[:,:,col])
                spk,  mem = self.lif1(fc1,mem)
                spk_rec.append(spk)
                mem_rec.append(mem)
        else:
            rate_coded = sgen.rate(x,1)[0]
            for row in range(x.shape[1]):
            
                fc1 = self.fc1(rate_coded[:,row,:])
                spk,  mem = self.lif1(fc1,mem)

            
        
                spk_rec.append(spk)
                mem_rec.append(mem)
            
        return torch.stack(spk_rec),torch.stack(mem_rec)

class SYNNETWORK(nn.Module):
    def __init__(self,mode="FULL"):
        super().__init__()

        if mode == "3D":
            self.fc1 = nn.Linear(100*100,3)
            self.timesteps = 200
        elif mode == "ROWS_COLUMNS":
            self.fc1 = nn.Linear(200,3)
            self.timesteps = 200
        else:
            self.fc1 = nn.Linear(100,3)

        self.leaky1 = snn.Leaky(0.95,1.0,learn_beta=True,learn_threshold=True)
        self.syn1 = CustomSynaptic(0.95,0.85,1.0,learn_alpha=True,learn_beta=True,learn_threshold=True)
        self.mode = mode
    
    def forward(self,x):
        syn, mem = self.syn1.reset_mem()
        mem1 = self.leaky1.init_leaky()
        spk_rec = []
        mem_rec = []

        if self.mode == "3D":
            for timestep in range(x.shape[-1]):
                input_data = x[:,:,:,timestep].flatten(start_dim=1)
                fc1 = self.fc1(input_data)

                spk, mem, syn = self.syn1(fc1, mem, syn)
                
                spk_rec.append(spk)
                mem_rec.append(mem)

        elif self.mode == "ROWS_COLUMNS":
            rate_coded = sgen.rate(x,1)[0]
            for index in range(x.shape[1]):
                row = rate_coded[:,index,:]
                col = rate_coded[:,:,index]
                row_col = torch.concatenate((row,col),dim=1)
                
                fc1 = self.fc1(row_col)
                spk, mem, syn = self.syn1(fc1, mem, syn)
                
                spk_rec.append(spk)
                mem_rec.append(mem)
        elif self.mode == "INTEGRATE_ROWS":
            rate_coded = sgen.rate(x,1)[0]
            for col in range(x.shape[1]):
            
                fc1 = self.fc1(rate_coded[:,:,col])
                spk,  mem, syn = self.syn1(fc1, mem, syn)

            
        
                spk_rec.append(spk)
                mem_rec.append(mem)
        else:
            rate_coded = sgen.rate(x,1)[0]
            for row in range(x.shape[1]):
            
                fc1 = self.fc1(rate_coded[:,row,:])
                spk, mem, syn = self.syn1(fc1, mem, syn)

            
        
                spk_rec.append(spk)
                mem_rec.append(mem)
            
        return torch.stack(spk_rec),torch.stack(mem_rec)

