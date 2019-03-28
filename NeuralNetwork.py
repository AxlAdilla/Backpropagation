from numpy import array,zeros
import math
import random
import json

class NeuralNetwork:
    def __init__(self,maxEpoh,learnRate,toleransiError):
        self.ErrOutput = [0]*4
        self.ErrHidden = [0]*10
        self.epoh=0
        self.maxEpoh = maxEpoh
        self.learnRate = learnRate
        self.numOfNeuronHidden = 10
        self.numOfNeuronOutput = 4
        self.numOfNeuronInput = 15
        self.z = [0]*11
        self.y = [0]*4
        self.x = [0]*16
        self.toleransiError = toleransiError
        self.AvgError = 1
        self.v = zeros((16,10))
        self.delta_v = zeros((16,10))
        self.w = zeros((11,4))
        self.delta_w = zeros((11,4))
        self.dataPelatihan=array([[0,1,0,
                                   1,1,0,
                                   0,1,0,
                                   0,1,0,
                                   1,1,1],
                                  [0,1,0,
                                   1,1,0,
                                   0,1,0,
                                   0,1,0,
                                   0,1,0],
                                  [0,1,0,
                                   0,1,0,
                                   0,1,0,
                                   0,1,0,
                                   0,1,0],
                                  [1,1,1,
                                   0,0,1,
                                   1,1,1,
                                   1,0,0,
                                   1,1,1],
                                  [0,1,1,
                                   1,0,1,
                                   0,1,0,
                                   1,0,0,
                                   1,1,1],
                                  [1,1,1,
                                   0,0,1,
                                   1,1,1,
                                   0,0,1,
                                   1,1,1],
                                  [1,1,0,
                                   0,0,1,
                                   1,1,0,
                                   0,0,1,
                                   1,1,0],
                                  [1,1,1,
                                   1,0,1,
                                   1,0,1,
                                   1,0,1,
                                   1,1,1],
                                  [0,1,0,
                                   1,0,1,
                                   1,0,1,
                                   1,0,1,
                                   0,1,0]])
        self.targetPelatihan = array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0]
                                     ,[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]])
    
    def kondisi_berhenti(self):
        if self.epoh > self.maxEpoh:
            return True
        #elif self.AvgError < self.toleransiError:
        #    return True
        else:
            return False
    
    def randomBobot(self):
        for i in range(0,self.numOfNeuronInput+1):
            for j in range(0,self.numOfNeuronHidden):
                self.v[i][j] = random.uniform(-1,1)
        #print("Self V: ")
        #print(self.v)
        for i in range(0,self.numOfNeuronHidden+1):
            for j in range(0,self.numOfNeuronOutput):
                self.w[i][j] = random.uniform(-1,1)
        #print("Self W: ")
        #print(self.w)
    
    def feedforward(self,indexPembelajaran):
        #print("index pembelajaran: {}".format(indexPembelajaran))
        
        #Mengisi nZ[] 
        nZ = [0]*self.numOfNeuronHidden
        for i in range(0,self.numOfNeuronHidden):
           for j in range(0,self.numOfNeuronInput):
               nZ[i] += self.dataPelatihan[indexPembelajaran][j]*self.v[j][i]
           nZ[i] += self.v[self.numOfNeuronInput][i]    
           #print("nZ[{0}]: {1}\n".format(i,nZ[i]))
        
        #nZ[i] masing-masing sudah ketemu , cari nilai Zi = f(nZ[i])
        for i in range(0,self.numOfNeuronHidden):
            self.z[i] = 1/(1+(math.exp(-nZ[i])))
        
        #mengisi nY[]
        nY = [0]*self.numOfNeuronOutput
        for i in range(0,self.numOfNeuronOutput):
           for j in range(0,self.numOfNeuronHidden):
               nY[i] += self.z[j]*self.w[j][i]
           nY[i] += self.w[self.numOfNeuronHidden][i]
           
        #nY[i] masing-masing sudah ketemu , cari nilai Yi = f(nY[i])
        for i in range(0,self.numOfNeuronOutput):
            self.y[i] = 1/(1+(math.exp(-nY[i])))

        
    def backpropagation(self,indexPembelajaran):
        #error output
        err_output = [0]*self.numOfNeuronOutput
        for i in range(len(err_output)):
            err_output[i]=(self.targetPelatihan[indexPembelajaran][i]-self.y[i])*self.y[i]*(1-self.y[i])
        
        #delta W 
        for i in range(self.numOfNeuronOutput):
            for j in range(self.numOfNeuronHidden):
                self.delta_w[j][i] = self.learnRate*err_output[i]*self.z[j]
            self.delta_w[self.numOfNeuronHidden][i] = self.learnRate*err_output[i]
        
        #hitung sigmaHid[j] = err_out[i]*w[j][i]
        sigmoidHid = [0]*self.numOfNeuronHidden
        for i in range(self.numOfNeuronHidden):
            for j in range(self.numOfNeuronOutput):
                sigmoidHid[i] += err_output[j]*self.w[i][j]
        
        #hitung err_hidden
        err_hidden = [0]*self.numOfNeuronHidden
        for i in range(len(err_hidden)):
            err_hidden[i]=sigmoidHid[i]*self.z[i]*(1-self.z[i])
            
        #hitung delta V
        for i in range(self.numOfNeuronHidden):
            for j in range(self.numOfNeuronInput):
                self.delta_v[j][i] = self.learnRate*err_hidden[i]*self.dataPelatihan[indexPembelajaran][j]
            self.delta_v[self.numOfNeuronHidden][i] = self.learnRate*err_hidden[i]
                
    def ubahData(self):
        #perbarui v
        for i in range(self.numOfNeuronHidden):
            for j in range(self.numOfNeuronInput):
                self.v[j][i] = self.v[j][i] + self.delta_v[j][i]
            self.v[self.numOfNeuronInput][i] = self.v[self.numOfNeuronInput][i] + self.delta_v[self.numOfNeuronInput][i]
        
        #perbarui w
        for i in range(self.numOfNeuronOutput):
            for j in range(self.numOfNeuronHidden):
                self.w[j][i] = self.w[j][i] + self.delta_w[j][i]
            self.w[self.numOfNeuronHidden][i] = self.w[self.numOfNeuronHidden][i] + self.delta_w[self.numOfNeuronHidden][i]
        
    def hitungAvgError(self):
        pass
    
    def createFile(self):
        file = open('bobot.txt','w')
        data = {}
        data['w'] = []
        data['w'] = self.w.tolist()
        data['v'] = []
        data['v'] = self.v.tolist()
        json.dump(data,file)
        file.close()