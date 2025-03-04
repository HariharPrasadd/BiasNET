import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

class PolarizationSimulation:
    def __init__(self, num_agents=50, num_issues=5, max_steps=500):
        # Simulation parameters
        self.num_agents = num_agents
        self.num_issues = num_issues
        self.max_steps = max_steps
        
        # Initialize agents' beliefs: random values between -1 and 1
        # This is now done ONCE at the beginning
        self.beliefs = np.random.uniform(-1, 1, size=(num_agents, num_issues))
        
        # Initialize affinity matrix (all start at 0)
        self.affinity = np.zeros((num_agents, num_agents))
        
        # For tracking purposes
        self.step_count = 0
        self.correlation_history = []
        self.polarization_metric_history = []
        
    def belief_similarity(self, agent1, agent2):
        """Calculate similarity of beliefs between two agents"""
        return np.mean(self.beliefs[agent1] * self.beliefs[agent2])
    
    def update_affinities(self):
        """Update all pairwise affinities based on belief similarity"""
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                similarity = self.belief_similarity(i, j)
                # Affinity changes proportionally to belief similarity
                change = similarity * 0.05
                self.affinity[i, j] += change
                self.affinity[j, i] += change
                
                # Constrain affinity to range [-1, 1]
                self.affinity[i, j] = max(-1, min(1, self.affinity[i, j]))
                self.affinity[j, i] = self.affinity[i, j]
    
    def update_beliefs(self):
        """Update beliefs based on social influence only"""
        # Create a copy of current beliefs to update from
        new_beliefs = self.beliefs.copy()

        #declare influence of positive and negative beliefs for affinity with other agents
        positive_influence_rate = 0.05
        negative_influence_rate = -0.03
        
        for agent in range(self.num_agents):
            for issue in range(self.num_issues):
                # Calculate social influence from all other agents
                social_influence = 0
                total_influence_weight = 0
                
                for other_agent in range(self.num_agents):
                    if other_agent != agent:
                        # Positive affinity leads to belief convergence
                        # Negative affinity leads to belief divergence (polarization)
                        if self.affinity[agent, other_agent] > 0:
                            # Pull toward the other agent's belief
                            influence = positive_influence_rate * self.affinity[agent, other_agent] * \
                                      (self.beliefs[other_agent, issue] - self.beliefs[agent, issue])
                            social_influence += influence
                        elif self.affinity[agent, other_agent] < 0:
                            # Push away from the other agent's belief
                            influence = negative_influence_rate * self.affinity[agent, other_agent] * \
                                      (self.beliefs[other_agent, issue] - self.beliefs[agent, issue])
                            social_influence -= influence
                
                # Apply social influence
                new_beliefs[agent, issue] += social_influence
                
                # Constrain beliefs to range [-1, 1]
                new_beliefs[agent, issue] = max(-1, min(1, new_beliefs[agent, issue]))
        
        # Update beliefs all at once
        self.beliefs = new_beliefs
    
    def calculate_correlation_matrix(self):
        """Calculate correlation matrix between issues"""
        return np.corrcoef(self.beliefs.T)
    
    def calculate_polarization_metric(self):
        """Calculate a metric for overall polarization"""
        # Use the mean absolute value of issue correlations as a simple metric
        corr_matrix = self.calculate_correlation_matrix()
        # Take the upper triangle of the correlation matrix, excluding the diagonal
        upper_tri = corr_matrix[np.triu_indices(self.num_issues, k=1)]
        return np.mean(np.abs(upper_tri))
    
    def calculate_belief_distance(self):
        """Calculate average Euclidean distance between all agents' belief vectors"""
        total_distance = 0
        count = 0
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                distance = np.linalg.norm(self.beliefs[i] - self.beliefs[j])
                total_distance += distance
                count += 1
        return total_distance / count if count > 0 else 0
    
    def step(self):
        """Execute one simulation step"""
        if self.step_count < self.max_steps:
            self.update_beliefs()
            self.update_affinities()
            
            # Record metrics
            self.correlation_history.append(self.calculate_correlation_matrix())
            self.polarization_metric_history.append(self.calculate_polarization_metric())
            
            self.step_count += 1
            return True
        return False
    
    def run_animation(self):
        """Run the simulation with animation"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3)
        
        # Subplot for agent network
        ax_network = fig.add_subplot(gs[0, 0])
        
        # Subplot for beliefs heatmap
        ax_beliefs = fig.add_subplot(gs[0, 1])
        
        # Subplot for correlation matrix
        ax_corr = fig.add_subplot(gs[0, 2])
        
        # Subplot for polarization metric over time
        ax_polarization = fig.add_subplot(gs[1, :])
        
        # Create a network graph
        G = nx.Graph()
        for i in range(self.num_agents):
            G.add_node(i)
        
        # Define colormap for node colors based on belief in issue 0
        cmap_nodes = plt.cm.RdBu
        
        # For edge colors and widths based on affinity
        pos = nx.spring_layout(G, seed=42)  # Fixed layout
        
        # Initialize heatmaps
        belief_heatmap = ax_beliefs.imshow(self.beliefs, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        corr_heatmap = ax_corr.imshow(np.zeros((self.num_issues, self.num_issues)), cmap='RdBu', vmin=-1, vmax=1)
        
        # Set labels for heatmaps
        ax_beliefs.set_xlabel('Issues')
        ax_beliefs.set_ylabel('Agents')
        ax_beliefs.set_title('Belief Distribution')
        ax_beliefs.set_xticks(range(self.num_issues))
        ax_beliefs.set_xticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        
        ax_corr.set_title('Issue Correlation Matrix')
        ax_corr.set_xlabel('Red = positive correlation \nBlue = negative correlation')
        ax_corr.set_xticks(range(self.num_issues))
        ax_corr.set_yticks(range(self.num_issues))
        ax_corr.set_xticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        ax_corr.set_yticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        
        # Initialize polarization plot
        polarization_line, = ax_polarization.plot([], [], lw=2, label='Correlation of Beliefs')
        belief_distance_line, = ax_polarization.plot([], [], lw=2, linestyle='--', color='green', 
                                                   label='Avg. Belief Distance')
        ax_polarization.set_xlim(0, self.max_steps)
        ax_polarization.set_ylim(0, 2)  # Adjusted for both metrics
        ax_polarization.set_xlabel('Step')
        ax_polarization.set_ylabel('Polarization Metrics')
        ax_polarization.set_title('Polarization Over Time')
        ax_polarization.grid(True)
        ax_polarization.legend()
        
        # Colorbar for beliefs
        plt.colorbar(belief_heatmap, ax=ax_beliefs, label='Belief Strength')
        plt.colorbar(corr_heatmap, ax=ax_corr, label='Correlation')
        
        # Title for network
        ax_network.set_title('Agent Network')
        
        # Text for simulation step
        step_text = ax_network.text(0.05, 0.95, '', transform=ax_network.transAxes)
        
        # Track the average belief distance metric
        belief_distance_history = []
        
        def init():
            # Draw network with initial node colors
            nx.draw_networkx_nodes(G, pos, node_color=[0] * self.num_agents, 
                                  node_size=300, alpha=0.8, ax=ax_network, cmap=cmap_nodes, vmin=-1, vmax=1)
            edges = nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, ax=ax_network)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_network)
            
            polarization_line.set_data([], [])
            belief_distance_line.set_data([], [])
            step_text.set_text('')
            
            return [belief_heatmap, corr_heatmap, polarization_line, belief_distance_line, step_text]
        
        def update(frame):
            # Run a simulation step
            if not self.step():
                ani.event_source.stop()
                plt.close()
                return
            
            # Calculate belief distance for this step
            belief_distance = self.calculate_belief_distance()
            belief_distance_history.append(belief_distance)
            
            # Update network visualization
            ax_network.clear()
            ax_network.set_title('Agent Network')
            
            # Update the graph edges based on affinities
            G.clear()
            for i in range(self.num_agents):
                G.add_node(i)
            
            for i in range(self.num_agents):
                for j in range(i+1, self.num_agents):
                    if abs(self.affinity[i, j]) > 0.1:  # Only draw significant connections
                        G.add_edge(i, j, weight=abs(self.affinity[i, j]))
            
            # Color nodes based on belief in the first issue
            node_colors = self.beliefs[:, 0]
            
            # Calculate edge colors and widths
            edge_colors = []
            edge_widths = []
            
            for u, v in G.edges():
                affinity_val = self.affinity[u, v]
                # Blue for positive affinity, red for negative
                if affinity_val > 0:
                    edge_colors.append('blue')
                else:
                    edge_colors.append('red')
                edge_widths.append(2 * abs(affinity_val))
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=300, alpha=0.8, ax=ax_network, cmap=cmap_nodes, vmin=-1, vmax=1)
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7, ax=ax_network)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_network)
            
            # Update beliefs heatmap
            belief_heatmap.set_array(self.beliefs)
            
            # Update correlation matrix
            corr_matrix = self.calculate_correlation_matrix()
            corr_heatmap.set_array(corr_matrix)
            
            # Update polarization plot
            polarization_line.set_data(range(len(self.polarization_metric_history)), self.polarization_metric_history)
            belief_distance_line.set_data(range(len(belief_distance_history)), belief_distance_history)
            
            # Update step text
            step_text.set_text(f'Step: {self.step_count}')
            step_text.set_position((0.05, 0.95))
            ax_network.add_artist(step_text)
            
            return [belief_heatmap, corr_heatmap, polarization_line, belief_distance_line, step_text]
        
        ani = FuncAnimation(fig, update, frames=self.max_steps, 
                           init_func=init, blit=False, interval=10)
        plt.tight_layout()
        plt.show()
        
        return ani

# Create and run the simulation
simulation = PolarizationSimulation(num_agents=40, num_issues=5, max_steps=500)
animation = simulation.run_animation()