import torch.nn as nn

class AVERAGEDMODELS():
    def startaverage(startModels, num_clients, input_size, hidden_size1):
        k=0
        for parameters in startModels[0].parameters():
            if(k == 0):
                new_params1 = parameters
            else:
                new_params2 = parameters
            k =k+1
        # averaged weights of the startModels 
        k =0
        for i  in range(num_clients):
            k =0
            for parameters in  startModels[i].parameters():
                if(k == 0):
                    # if(flag ==1):
                    #     new_params1 = new_params1 + parameters*percentage_loss[i] 
                    new_params1 = new_params1 + parameters 
                else:
                    # if(flag ==1):
                    #     new_params2 = new_params2 + parameters*percentage_loss[i] 
                    new_params2 = new_params2 + parameters 
                k =k+1
        new_params1 =[number1/num_clients for number1 in new_params1] 
        new_params2 = [number2/num_clients for number2 in new_params2]

        class ClientStartNN(nn.Module):
            def __init__(self, input_size, hidden_size1):
                super(ClientStartNN, self).__init__()
                self.l1 = nn.Linear(input_size, hidden_size1)
                self.relu = nn.ReLU()

            def forward(self, x):
                out = self.l1(x)
                out = self.relu(out)
                return out

        k =0
        for sample in ClientStartNN(input_size, hidden_size1).parameters():
            if (k ==0):
                ClientStartNN.parameters = new_params1
            else:
                ClientStartNN.parameters = new_params2
            k =k+1
        return ClientStartNN

    def endaverage(endModels, num_clients,input_size, hidden_size2, num_classes):
        k=0
        for parameters in endModels[0].parameters():
            if(k == 0):
                new_params3 = parameters
            else:
                new_params4 = parameters
            k =k+1
        # averaged weights of the endModels 
        k =0
        for i  in range(num_clients):
            k =0
            for parameters in  endModels[i].parameters():
                if(k == 0):
                    # if(flag ==1):
                    #     new_params3 = new_params3 + parameters*percentage_loss[i] 
                    new_params3 = new_params3 + parameters
                else:
                    # if(flag == 0):
                    #     new_params4 = new_params4 + parameters*percentage_loss[i]
                    new_params4 = new_params4 + parameters
                k = k+1
        new_params3 =[number3/num_clients for number3 in new_params3]
        new_params4 = [number4/num_clients for number4 in new_params4]

        class ClientEndNN(nn.Module):
            def __init__(self, hidden_size2, num_classes):
                super(ClientEndNN, self).__init__()
                self.input_size = input_size
                self.l1 = nn.Linear(hidden_size2, num_classes)
                
            def forward(self, x):
                out = self.l1(x)
                return out
        k =0
        for sample in ClientEndNN(hidden_size2, num_classes).parameters():
            if (k ==0):
                ClientEndNN.parameters = new_params3
            else:
                ClientEndNN.parameters = new_params4
            k= k+1
        return ClientEndNN

    def weightedaverage_loss():


        return model

    def algo1():
        return model
    
    def algo2():
        return model