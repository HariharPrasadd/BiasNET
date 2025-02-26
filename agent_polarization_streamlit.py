import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
import time

class PolarizationSimulation:
    def __init__(self, num_agents=40, num_issues=5, max_steps=500, 
                 affinity_change_rate=0.05, positive_influence_rate=0.05, 
                 negative_influence_rate=0.03):
        # Simulation parameters
        self.num_agents = num_agents
        self.num_issues = num_issues
        self.max_steps = max_steps
        self.affinity_change_rate = affinity_change_rate
        self.positive_influence_rate = positive_influence_rate
        self.negative_influence_rate = negative_influence_rate
        
        # Initialize agents' beliefs: random values between -1 and 1
        self.beliefs = np.random.uniform(-1, 1, size=(num_agents, num_issues))
        
        # Initialize affinity matrix (all start at 0)
        self.affinity = np.zeros((num_agents, num_agents))
        
        # For tracking purposes
        self.step_count = 0
        self.correlation_history = []
        self.polarization_metric_history = []
        self.belief_distance_history = []
        
    def belief_similarity(self, agent1, agent2):
        """Calculate similarity of beliefs between two agents"""
        return np.mean(self.beliefs[agent1] * self.beliefs[agent2])
    
    def update_affinities(self):
        """Update all pairwise affinities based on belief similarity"""
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                similarity = self.belief_similarity(i, j)
                # Affinity changes proportionally to belief similarity
                change = similarity * self.affinity_change_rate
                self.affinity[i, j] += change
                self.affinity[j, i] += change
                
                # Constrain affinity to range [-1, 1]
                self.affinity[i, j] = max(-1, min(1, self.affinity[i, j]))
                self.affinity[j, i] = self.affinity[i, j]
    
    def update_beliefs(self):
        """Update beliefs based on social influence only"""
        # Create a copy of current beliefs to update from
        new_beliefs = self.beliefs.copy()
        
        for agent in range(self.num_agents):
            for issue in range(self.num_issues):
                # Calculate social influence from all other agents
                social_influence = 0
                
                for other_agent in range(self.num_agents):
                    if other_agent != agent:
                        # Positive affinity leads to belief convergence
                        # Negative affinity leads to belief divergence (polarization)
                        if self.affinity[agent, other_agent] > 0:
                            # Pull toward the other agent's belief
                            influence = self.positive_influence_rate * self.affinity[agent, other_agent] * \
                                      (self.beliefs[other_agent, issue] - self.beliefs[agent, issue])
                            social_influence += influence
                        elif self.affinity[agent, other_agent] < 0:
                            # Push away from the other agent's belief
                            influence = self.negative_influence_rate * self.affinity[agent, other_agent] * \
                                      (self.beliefs[other_agent, issue] - self.beliefs[agent, issue])
                            social_influence += influence
                
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
            self.belief_distance_history.append(self.calculate_belief_distance())
            
            self.step_count += 1
            return True
        return False
    
    def create_visualization(self):
        """Create visualization for the current state"""
        fig = plt.figure(figsize=(15, 10))
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
        
        # Add edges based on affinities
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if abs(self.affinity[i, j]) > 0.1:  # Only draw significant connections
                    G.add_edge(i, j, weight=abs(self.affinity[i, j]))
        
        # Define layout
        pos = nx.spring_layout(G, seed=42)
        
        # Node colors based on belief in the first issue
        node_colors = self.beliefs[:, 0]
        
        # Edge colors and widths
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            affinity_val = self.affinity[u, v]
            if affinity_val > 0:
                edge_colors.append('blue')
            else:
                edge_colors.append('red')
            edge_widths.append(2 * abs(affinity_val))
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=300, alpha=0.8, ax=ax_network, 
                              cmap=plt.cm.RdBu, vmin=-1, vmax=1)
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              edge_color=edge_colors, alpha=0.7, ax=ax_network)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax_network)
        ax_network.set_title('Agent Network')
        
        # Draw beliefs heatmap
        belief_heatmap = ax_beliefs.imshow(self.beliefs, cmap='RdBu', 
                                         vmin=-1, vmax=1, aspect='auto')
        ax_beliefs.set_xlabel('Issues')
        ax_beliefs.set_ylabel('Agents')
        ax_beliefs.set_title('Belief Distribution')
        ax_beliefs.set_xticks(range(self.num_issues))
        ax_beliefs.set_xticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        plt.colorbar(belief_heatmap, ax=ax_beliefs, label='Belief Strength')
        
        # Draw correlation matrix
        corr_matrix = self.calculate_correlation_matrix()
        corr_heatmap = ax_corr.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax_corr.set_title('Issue Correlation Matrix')
        ax_corr.set_xlabel('Red = positive correlation \nBlue = negative correlation')
        ax_corr.set_xticks(range(self.num_issues))
        ax_corr.set_yticks(range(self.num_issues))
        ax_corr.set_xticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        ax_corr.set_yticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        plt.colorbar(corr_heatmap, ax=ax_corr, label='Correlation')
        
        # Draw polarization plot
        ax_polarization.plot(self.polarization_metric_history, lw=2, label='Correlation of Beliefs')
        ax_polarization.plot(self.belief_distance_history, lw=2, linestyle='--', 
                           color='green', label='Avg. Belief Distance')
        ax_polarization.set_xlim(0, self.max_steps)
        ax_polarization.set_ylim(0, 2)  # Adjusted for both metrics
        ax_polarization.set_xlabel('Step')
        ax_polarization.set_ylabel('Polarization Metrics')
        ax_polarization.set_title('Polarization Over Time')
        ax_polarization.grid(True)
        ax_polarization.legend()
        
        # Add step text
        plt.figtext(0.02, 0.02, f'Step: {self.step_count}', fontsize=12)
        
        plt.tight_layout()
        return fig

# Streamlit app
def main():
    st.title("Social Polarization Simulation")
    st.write("""
    This application simulates the emergence of polarization in social networks.
    Agents hold beliefs on multiple issues and develop affinities with other agents based on belief similarity.
    Positive affinity leads to belief convergence, while negative affinity leads to belief divergence.
    """)
    
    st.header("Simulation Parameters")
    
    # Sidebar for parameters
    with st.sidebar:
        st.subheader("Simulation Settings")
        num_agents = st.slider("Number of agents", 10, 100, 40, 
                             help="Number of agents in the simulation")
        num_issues = st.slider("Number of belief dimensions", 2, 10, 5, 
                             help="Number of issues that agents have beliefs about")
        affinity_change_rate = st.slider("Affinity change rate", 0.01, 0.2, 0.05, 0.01, 
                                       help="Scaling factor for affinity updates")
        positive_influence_rate = st.slider("Positive influence strength", 0.01, 0.2, 0.05, 0.01, 
                                          help="Strength of positive influence")
        negative_influence_rate = st.slider("Negative influence strength", 0.01, 0.2, 0.03, 0.01, 
                                          help="Strength of negative influence")
        max_steps = st.slider("Maximum simulation steps", 100, 1000, 500, 50)
        
        step_increment = st.slider("Steps per update", 1, 20, 5, 
                                 help="Number of simulation steps per visualization update")
    
    # Create simulation with user parameters
    if 'simulation' not in st.session_state or st.button("Reset Simulation"):
        st.session_state.simulation = PolarizationSimulation(
            num_agents=num_agents,
            num_issues=num_issues,
            max_steps=max_steps,
            affinity_change_rate=affinity_change_rate,
            positive_influence_rate=positive_influence_rate,
            negative_influence_rate=negative_influence_rate
        )
        st.session_state.paused = True
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Run/Resume"):
            st.session_state.paused = False
    with col2:
        if st.button("Pause"):
            st.session_state.paused = True
    with col3:
        if st.button("Step"):
            sim = st.session_state.simulation
            for _ in range(step_increment):
                if not sim.step():
                    break
    
    # Create a placeholder for the simulation visualization
    vis_placeholder = st.empty()
    
    # Display current visualization
    vis_placeholder.pyplot(st.session_state.simulation.create_visualization())
    
    # Run simulation if not paused
    if not st.session_state.paused:
        sim = st.session_state.simulation
        progress_bar = st.progress(0)
        
        # Run simulation steps
        for i in range(step_increment):
            if not sim.step():
                st.session_state.paused = True
                break
            progress_bar.progress(sim.step_count / sim.max_steps)
        
        # Update visualization
        vis_placeholder.pyplot(sim.create_visualization())
        
        # Rerun to continue simulation
        time.sleep(0.1)  # Small delay to prevent too rapid updates
        if not st.session_state.paused:
            st.experimental_rerun()
    
    # Display metrics
    st.subheader("Current Metrics")
    col1, col2 = st.columns(2)
    with col1:
        sim = st.session_state.simulation
        if len(sim.polarization_metric_history) > 0:
            st.metric("Belief Correlation", 
                    round(sim.polarization_metric_history[-1], 3))
    with col2:
        if len(sim.belief_distance_history) > 0:
            st.metric("Avg. Belief Distance", 
                    round(sim.belief_distance_history[-1], 3))

if __name__ == "__main__":
    main()