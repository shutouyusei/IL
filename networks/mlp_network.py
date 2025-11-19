import torch
import torch.nn as nn
import torchvision.models as models

class MlpNetwork(nn.Module):
    def __init__(self, state_input_dim, action_output_dim, cnn_features_dim=512, mlp_hidden_dim=128):
        super().__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.num_actions = action_output_dim
        self.num_classes = 3
        
        self.state_mlp = nn.Sequential(
            nn.Linear(state_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        )
        
        combined_input_dim = cnn_features_dim + mlp_hidden_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, action_output_dim*self.num_classes)
        )

    def forward(self, image, state):
        img_features = self.cnn_backbone(image)
        # (B, 512, 1, 1) -> (B, 512)
        img_features = torch.flatten(img_features, 1) 

        state_features = self.state_mlp(state)
        
        # (B, 512) + (B, 128) -> (B, 640)
        combined_features = torch.cat((img_features, state_features), dim=1)
        
        output_action = self.fusion_head(combined_features)

        output_action = output_action.view(output_action.size(0),self.num_actions,self.num_classes)
        output_action = output_action.permute(0,2,1)
        
        return output_action
