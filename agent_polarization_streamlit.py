import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
import time
from functools import lru_cache

class PolarizationSimulation:
    def __init__(self, num_agents=40, num_issues=5, max_steps=500, 
                 affinity_change_rate=0.05, positive_influence_rate=0.05, 
                 negative_influence_rate=-0.03):
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
        
        # Pre-allocate arrays for performance
        self.new_beliefs = np.zeros_like(self.beliefs)
        
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
        # Use pre-allocated array for speed
        np.copyto(self.new_beliefs, self.beliefs)
        
        for agent in range(self.num_agents):
            for issue in range(self.num_issues):
                # Calculate social influence from all other agents
                social_influence = 0
                
                for other_agent in range(self.num_agents):
                    if other_agent != agent:
                        aff = self.affinity[agent, other_agent]
                        diff = self.beliefs[other_agent, issue] - self.beliefs[agent, issue]
                        
                        if aff > 0:
                            # Pull toward the other agent's belief
                            influence = self.positive_influence_rate * aff * diff
                        elif aff < 0:
                            # Push away from the other agent's belief
                            influence = self.negative_influence_rate * aff * diff
                        else:
                            continue  # Skip if affinity is 0
                            
                        social_influence += influence
                
                # Apply social influence
                self.new_beliefs[agent, issue] += social_influence
                
                # Constrain beliefs to range [-1, 1]
                self.new_beliefs[agent, issue] = max(-1, min(1, self.new_beliefs[agent, issue]))
        
        # Swap arrays instead of copying
        self.beliefs, self.new_beliefs = self.new_beliefs, self.beliefs
    
    def calculate_correlation_matrix(self):
        """Calculate correlation matrix between issues"""
        return np.corrcoef(self.beliefs.T)
    
    def calculate_polarization_metric(self):
        """Calculate a metric for overall polarization"""
        corr_matrix = self.calculate_correlation_matrix()
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
    
    def run_multiple_steps(self, num_steps):
        """Run multiple simulation steps efficiently"""
        if self.step_count >= self.max_steps:
            return False
            
        steps_to_run = min(num_steps, self.max_steps - self.step_count)
        
        for _ in range(steps_to_run):
            self.update_beliefs()
            self.update_affinities()
            self.step_count += 1
            
        # Only calculate and record metrics after all steps
        self.correlation_history.append(self.calculate_correlation_matrix())
        self.polarization_metric_history.append(self.calculate_polarization_metric())
        self.belief_distance_history.append(self.calculate_belief_distance())
        
        return True
    
    def step(self):
        """Execute one simulation step"""
        return self.run_multiple_steps(1)
    
    def create_visualization(self):
        """Create visualization for the current state"""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3)
        
        # Subplot for agent network
        ax_network = fig.add_subplot(gs[0, 0])
        
        # Create network only for agents with significant connections for efficiency
        G = nx.Graph()
        for i in range(self.num_agents):
            G.add_node(i)
        
        # Add edges based on affinities
        edge_colors = []
        edge_widths = []
        
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if abs(self.affinity[i, j]) > 0.1:  # Only draw significant connections
                    G.add_edge(i, j, weight=abs(self.affinity[i, j]))
                    
                    affinity_val = self.affinity[i, j]
                    if affinity_val > 0:
                        edge_colors.append('blue')
                    else:
                        edge_colors.append('red')
                    edge_widths.append(2 * abs(affinity_val))
        
        # Define layout - use cached layout for speed if available
        if not hasattr(self, 'pos'):
            self.pos = nx.spring_layout(G, seed=42)
        
        # Node colors based on belief in the first issue
        node_colors = self.beliefs[:, 0]
        
        # Draw the network
        nx.draw_networkx_nodes(G, self.pos, node_color=node_colors, 
                              node_size=300, alpha=0.8, ax=ax_network, 
                              cmap=plt.cm.RdBu, vmin=-1, vmax=1)
        nx.draw_networkx_edges(G, self.pos, width=edge_widths, 
                              edge_color=edge_colors, alpha=0.7, ax=ax_network)
        nx.draw_networkx_labels(G, self.pos, font_size=8, ax=ax_network)
        ax_network.set_title('Agent Network')
        
        # Subplot for beliefs heatmap
        ax_beliefs = fig.add_subplot(gs[0, 1])
        belief_heatmap = ax_beliefs.imshow(self.beliefs, cmap='RdBu', 
                                         vmin=-1, vmax=1, aspect='auto')
        ax_beliefs.set_xlabel('Issues')
        ax_beliefs.set_ylabel('Agents')
        ax_beliefs.set_title('Belief Distribution')
        ax_beliefs.set_xticks(range(self.num_issues))
        ax_beliefs.set_xticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        plt.colorbar(belief_heatmap, ax=ax_beliefs, label='Belief Strength')
        
        # Subplot for correlation matrix
        ax_corr = fig.add_subplot(gs[0, 2])
        corr_matrix = self.calculate_correlation_matrix()
        corr_heatmap = ax_corr.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax_corr.set_title('Issue Correlation Matrix')
        ax_corr.set_xlabel('Red = positive correlation \nBlue = negative correlation')
        ax_corr.set_xticks(range(self.num_issues))
        ax_corr.set_yticks(range(self.num_issues))
        ax_corr.set_xticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        ax_corr.set_yticklabels([f'Issue {i+1}' for i in range(self.num_issues)])
        plt.colorbar(corr_heatmap, ax=ax_corr, label='Correlation')
        
        # Subplot for polarization metrics over time
        ax_polarization = fig.add_subplot(gs[1, :])
        
        # Only plot points that exist (for efficiency)
        x_points = list(range(len(self.polarization_metric_history)))
        
        # Draw lines efficiently by using vectorized operations
        ax_polarization.plot(x_points, self.polarization_metric_history, lw=2, label='Correlation of Beliefs')
        ax_polarization.plot(x_points, self.belief_distance_history, lw=2, linestyle='--', 
                           color='green', label='Avg. Belief Distance')
        
        # Set plot limits and labels
        ax_polarization.set_xlim(0, self.max_steps)
        ax_polarization.set_ylim(0, 2)
        ax_polarization.set_xlabel('Step')
        ax_polarization.set_ylabel('Polarization Metrics')
        ax_polarization.set_title('Polarization Over Time')
        ax_polarization.grid(True)
        ax_polarization.legend()
        
        # Add step text
        plt.figtext(0.02, 0.02, f'Step: {self.step_count}', fontsize=12)
        
        plt.tight_layout()
        return fig


# Streamlit app with optimizations
def main():
    # Set page config for faster loading
    st.set_page_config(
        page_title="Social Polarization Simulation",
        layout="wide",
    )
    
    st.title("Social Polarization Simulation")
    st.write("""
    This application simulates the emergence of polarization in social networks.
    Agents hold beliefs on multiple issues and develop affinities with other agents based on belief similarity.
    """)
    
    # Sidebar for parameters with cached state
    with st.sidebar:
        st.subheader("Simulation Settings")
        
        # Only show these controls if simulation hasn't started or if reset is clicked
        num_agents = st.slider("Number of agents", 10, 100, 40)
        num_issues = st.slider("Number of belief dimensions", 2, 10, 5)
        affinity_change_rate = st.slider("Affinity change rate", 0.01, 0.2, 0.05, 0.01)
        positive_influence_rate = st.slider("Positive influence strength", 0.01, 0.2, 0.05, 0.01)
        negative_influence_rate = st.slider("Negative influence strength", -0.2, -0.01, -0.03, 0.01)
        max_steps = st.slider("Maximum simulation steps", 100, 1000, 500, 50)
        
        # Performance settings
        st.subheader("Performance Settings")
        step_increment = st.slider("Steps per update", 1, 50, 10, 
                                 help="Higher values = faster simulation but less smooth animation")
        update_interval = st.slider("Update interval (ms)", 10, 1000, 100, 10,
                                  help="Time between updates (lower = faster but may strain CPU)")
    
    # Initialize session state for simulation if not already present
    if 'simulation' not in st.session_state or st.sidebar.button("Reset Simulation", use_container_width=True):
        st.session_state.simulation = PolarizationSimulation(
            num_agents=num_agents,
            num_issues=num_issues,
            max_steps=max_steps,
            affinity_change_rate=affinity_change_rate,
            positive_influence_rate=positive_influence_rate,
            negative_influence_rate=negative_influence_rate
        )
        st.session_state.paused = True
        st.session_state.last_update_time = time.time()
    
    # Create more efficient control layout
    controls = st.container()
    col1, col2, col3 = controls.columns(3)
    
    with col1:
        run_button = st.button("Run/Resume", use_container_width=True)
        if run_button:
            st.session_state.paused = False
    
    with col2:
        pause_button = st.button("Pause", use_container_width=True)
        if pause_button:
            st.session_state.paused = True
    
    with col3:
        step_button = st.button("Step", use_container_width=True)
        if step_button:
            sim = st.session_state.simulation
            sim.run_multiple_steps(step_increment)
    
    # Create placeholder for progress bar and metrics
    progress_container = st.container()
    metrics_container = st.container()
    
    # Create figure placeholder to avoid recreation
    fig_placeholder = st.empty()
    
    # Run simulation loop more efficiently
    sim = st.session_state.simulation
    current_time = time.time()
    
    # Display metrics more efficiently
    with metrics_container:
        st.subheader("Current Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            if len(sim.polarization_metric_history) > 0:
                st.metric("Belief Correlation", 
                        round(sim.polarization_metric_history[-1], 3))
        with col2:
            if len(sim.belief_distance_history) > 0:
                st.metric("Avg. Belief Distance", 
                        round(sim.belief_distance_history[-1], 3))
    
    # Display progress bar
    with progress_container:
        progress = st.progress(sim.step_count / sim.max_steps)
    
    # Only update visualization at appropriate intervals
    fig_placeholder.pyplot(sim.create_visualization())
    
    # Efficient simulation loop when not paused
    if not st.session_state.paused:
        if (current_time - st.session_state.last_update_time) * 1000 >= update_interval:
            if sim.run_multiple_steps(step_increment):
                progress.progress(sim.step_count / sim.max_steps)
                st.session_state.last_update_time = current_time
                time.sleep(update_interval / 1000)  # Controlled delay
                st.rerun()
            else:
                st.session_state.paused = True
                st.info("Simulation completed!")
        else:
            time.sleep(0.01)  # Small delay to prevent CPU overuse
            st.rerun()

if __name__ == "__main__":
    main()