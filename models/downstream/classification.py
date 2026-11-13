import torch
import torch.nn as nn

from sklearn.preprocessing import LabelBinarizer

class LinearClassificationHead(nn.Module):
    def __init__(self, 
                 input_dim=512, 
                 n_classes=4, 
                 device:torch.device="cpu"):
        
        super().__init__()
        self.device = device

        self.fc = nn.Linear(input_dim, n_classes)

        # One-hot labelizer
        self.lb = LabelBinarizer()
        # Loss function
        self.loss_function = torch.nn.CrossEntropyLoss()
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fit_labelizer(self,labels):
        self.lb.fit(labels)

    def transform_labels(self, labels):
        targets = torch.tensor(self.lb.transform(labels), dtype=torch.float32)
        targets = targets.to(self.device, dtype=torch.float32)
        return targets

    def forward(self, x):
        return self.fc(x)
    
    def compute_loss(self, outputs, inputs):
        y_label = inputs['y_label']

        targets = torch.argmax(self.transform_labels(y_label), dim=-1)
        predictions = outputs.to(torch.float32)  # Ensure predictions are float32
        
        return self.loss_function(predictions, targets)
class MLPClassificationHead(nn.Module):
    def __init__(self, 
                 in_dim=512, 
                 n_classes=4, 
                 dropout:float=0.5,
                 device:torch.device="cpu"):
        
        super().__init__()
        self.device = device

        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, n_classes),
            nn.Dropout(dropout)
        )

        # One-hot labelizer
        self.lb = LabelBinarizer()
        # Loss function
        self.loss_function = torch.nn.CrossEntropyLoss()
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fit_labelizer(self,labels):
        self.lb.fit(labels)
    
    def transform_labels(self, labels):
        targets = torch.tensor(self.lb.transform(labels), dtype=torch.float32)
        targets = targets.to(self.device, dtype=torch.float32)
        return targets

    def forward(self, x):
        return self.fc(x)
    
    def compute_loss(self, outputs, inputs):
        y_label = inputs['y_label']

        targets = torch.argmax(self.transform_labels(y_label), dim=-1)
        predictions = outputs.to(torch.float32)  # Ensure predictions are float32
        
        return self.loss_function(predictions, targets)
    
class OldClassificationHead(nn.Module):
    def __init__(self, 
                 in_dim=512, 
                 n_classes=4, 
                 dropout:float=0.5,
                 device:torch.device="cpu"):
        
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim//2, n_classes)
        )

        self.lb = LabelBinarizer()
        self.device = device

        self.loss_function = torch.nn.CrossEntropyLoss()

    def fit_labelizer(self,labels):
        self.lb.fit(labels)
    
    def transform_labels(self, labels):
        targets = torch.tensor(self.lb.transform(labels), dtype=torch.float32)
        targets = targets.to(self.device, dtype=torch.float32)
        return targets

    def forward(self, x):
        return self.fc(x)
    
    def compute_loss(self, outputs, inputs):
        y_label = inputs['y_label']

        targets = torch.argmax(self.transform_labels(y_label), dim=-1)
        predictions = outputs.to(torch.float32)  # Ensure predictions are float32
        
        return self.loss_function(predictions, targets)