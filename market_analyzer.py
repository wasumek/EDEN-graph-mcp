import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

class MarketAnalyzer:
    def __init__(self, file_path, sheet_name='Sheet1'):
        self.df = pd.read_excel(file_path, sheet_name=sheet_name)
        # Anonymize Organization Names (Use 'Market segment' + Index)
        if 'Organization Name CB' in self.df.columns and 'Market segment' in self.df.columns:
             self.df['Organization Name CB'] = [f"{row['Market segment']} {i+1}" for i, row in self.df.iterrows()]
        self.financial_columns = [
            'Found Year',
            'Number of Investors',
            'Total Funding Raised (in millions)',
            'Number of Employees',
            'IT Spend',
            'Patents Granted',
            'Estimated Revenue Range'
        ]
        self.remapped_labels = {
            'Number of Investors': 'Investors',
            'Number of Employees': 'Employees',
            'Total Funding Raised (in millions)': 'Total Raised',
            'Estimated Revenue Range': 'Est. Revenue'
        }
        self.cluster_categories = {
            1: 'Clinical Software',
            2: 'Non-Health Systems',
            3: 'Monitoring & Diagnostics', 
            4: 'Digital Therapeutics',
            5: 'Health & Wellness',
            6: 'Care Support'
        }
        
    def collect_user_data(self, market_segment, product, customer, founded_year, 
                        investors, funding, employees):
        """Collect user input data from parameters"""
        try:
            user_data = {
                'Organization Name CB': 'You',
                'Market segment': market_segment,
                'Value object 1': product,
                'To whom 1': customer,
                'Found Year': 2024 - int(founded_year),
                'Number of Investors': int(investors),
                'Total Funding Raised (in millions)': float(funding),
                'Number of Employees': int(employees),
                'IT Spend': 50000,
                'Patents Granted': 1,
                'Estimated Revenue Range': 10,
                'Crunchbase Rank': 1e6
            }
            self.df = pd.concat([self.df, pd.DataFrame([user_data])], ignore_index=True)
        except ValueError as e:
            raise ValueError(f"Invalid input: {str(e)}")

    def perform_clustering(self, max_d=1.5):
        """Perform hierarchical clustering on the dataset"""
        df_VP = pd.DataFrame(self.df[['Market segment', 'Value object 1', 'To whom 1']])
        df_encoded = pd.get_dummies(df_VP)
        
        distance_matrix = pdist(df_encoded, metric='jaccard')
        self.linkage_matrix = linkage(distance_matrix, method='ward')
        
        cluster_labels = fcluster(self.linkage_matrix, max_d, criterion='distance')
        self.df['Cluster Label'] = pd.Series(cluster_labels).map(self.cluster_categories)

    def plot_dendrogram(self):
        """Plot hierarchical clustering dendrogram using Plotly"""
        try:
            labels = self.df[['Market segment', 'Value object 1', 'To whom 1']].apply(
                lambda row: ' | '.join(row.values.astype(str)), axis=1
            ).tolist()

            # Create dendrogram figure using figure factory
            fig = ff.create_dendrogram(
                X=self.linkage_matrix,
                orientation='right',
                labels=labels,
                distfun=None,  # Distance function already applied
                linkagefun=lambda x: self.linkage_matrix,  # Use pre-computed linkage
                color_threshold=1.5  # Match max_d from perform_clustering
            )

            # Update layout
            fig.update_layout(
                title={
                    'text': 'Market Segmentation Dendrogram',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=16)
                },
                width=800,
                height=800,
                showlegend=False,
                xaxis_title="Distance",
                yaxis_title="Companies",
                yaxis=dict(
                    side='right',  # Move labels to right side
                    showgrid=False
                    ),
                font=dict(size=10)
            )

            # Return the figure instead of showing it
            return fig
        except Exception as e:
            print(f"Error generating dendrogram: {str(e)}")
            return None

    def plot_cluster_overview(self):
        """Plot overview of all clusters' financial metrics using Plotly"""
        try:
            # Remove outliers using Z-score method
            def remove_outliers_zscore(df, columns, threshold=3):
                df_clean = df.copy()
                for col in columns:
                    if col != 'Estimated Revenue Range':
                        z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                        df_clean = df_clean[(z_scores < threshold).reindex(df_clean.index, fill_value=False)]
                return df_clean

            df_no_outliers = remove_outliers_zscore(self.df, self.financial_columns, threshold=3)

            # Create subplot layout
            fig = make_subplots(
                rows=3, 
                cols=3,
                subplot_titles=[self.remapped_labels.get(metric, metric) 
                              for metric in self.financial_columns],
                vertical_spacing=0.12,
                horizontal_spacing=0.05
            )

            # Define color palette
            colors = px.colors.qualitative.Set3

            # Create box plots for each metric
            for i, metric in enumerate(self.financial_columns):
                row = i // 3 + 1
                col = i % 3 + 1

                # Add box plot
                fig.add_trace(
                    go.Box(
                        x=df_no_outliers['Cluster Label'],
                        y=df_no_outliers[metric],
                        name=metric,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8,
                        marker=dict(
                            opacity=0.7
                        ),
                        line=dict(color=colors[i % len(colors)]),
                        showlegend=False,
                        hovertemplate=metric + ': %{y}<extra></extra>'
                    ),
                    row=row,
                    col=col
                )

                # Highlight user's position if available
                if 'You' in df_no_outliers['Organization Name CB'].values:
                    user_value = df_no_outliers[df_no_outliers['Organization Name CB'] == 'You'][metric].iloc[0]
                    user_cluster = df_no_outliers[df_no_outliers['Organization Name CB'] == 'You']['Cluster Label'].iloc[0]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[user_cluster],
                            y=[user_value],
                            mode='markers',
                            marker=dict(
                                symbol='star',
                                size=12,
                                color='red',
                                line=dict(color='red', width=1)
                            ),
                            name='Your Position',
                            showlegend=False,
                            hovertemplate='Your value: %{y}<extra></extra>'
                        ),
                        row=row,
                        col=col
                    )

                # Add median annotations
                for cluster in df_no_outliers['Cluster Label'].unique():
                    median = df_no_outliers[df_no_outliers['Cluster Label'] == cluster][metric].median()
                    fig.add_annotation(
                        x=cluster,
                        y=median,
                        text=f'{median:,.0f}',
                        showarrow=False,
                        font=dict(size=8),
                        yshift=10,
                        row=row,
                        col=col
                    )

            # Update layout
            fig.update_layout(
                title=dict(
                    text='Market Segment Comparison Across Financial Metrics',
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=16)
                ),
                showlegend=False,
                height=1000,
                width=1200,
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(family="Arial", size=10),
                margin=dict(t=100, b=50, l=50, r=50)
            )

            # Update axes
            fig.update_xaxes(
                tickangle=45,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=8)
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickfont=dict(size=8)
            )

            return fig

        except Exception as e:
            print(f"Error generating cluster overview: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    def plot_cluster_comparison(self):
        """Plot comparison of user's position within their cluster using Plotly"""
        try:
            user_cluster = self.df[self.df['Organization Name CB'] == 'You']['Cluster Label'].iloc[0]
            cluster_df = self.df[self.df['Cluster Label'] == user_cluster].copy()

            # Create subplot layout with more space
            fig = make_subplots(
                rows=1,
                cols=len(self.financial_columns),
                subplot_titles=[self.remapped_labels.get(metric, metric) 
                            for metric in self.financial_columns],
                horizontal_spacing=0.08
            )

            # Create box plots for each metric
            for i, metric in enumerate(self.financial_columns):
                # Create a constant x-value array for the box plot
                x_vals = ['Cluster'] * len(cluster_df[metric])
                
                # Add box plot
                fig.add_trace(
                    go.Box(
                        x=x_vals,  # Use constant x values
                        y=cluster_df[metric],
                        name=metric,
                        boxpoints='all',  # Show all points
                        jitter=0.5,  # Add more jitter for better point distribution
                        pointpos=-1.5,  # Adjust point position
                        marker=dict(
                            color='#DDCB5B',
                            size=6,
                            opacity=0.7
                        ),
                        line=dict(
                            color='#DDCB5B',
                            width=2
                        ),
                        showlegend=False,
                        hoverinfo='y'
                    ),
                    row=1,
                    col=i+1
                )

                # Add user's position
                user_value = self.df[self.df['Organization Name CB'] == 'You'][metric].iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=['Cluster'],
                        y=[user_value],
                        mode='markers',
                        marker=dict(
                            symbol='star',
                            size=15,
                            color='red',
                            line=dict(color='red', width=2)
                        ),
                        name='Your Position',
                        showlegend=False,
                        hovertemplate='Your value: %{y}<extra></extra>'
                    ),
                    row=1,
                    col=i+1
                )

                # Add median value
                median_val = np.ceil(cluster_df[metric].median())
                fig.add_annotation(
                    x='Cluster',
                    y=median_val,
                    text=f'Median: {median_val:,.0f}',
                    showarrow=False,
                    font=dict(size=10),
                    yshift=10,
                    row=1,
                    col=i+1
                )

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'Your Position in {user_cluster} Segment',
                    x=0.5,
                    y=0.98,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=16)
                ),
                showlegend=False,
                height=500,
                width=1500,
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(family="Arial", size=10),
                margin=dict(t=80, b=50, l=50, r=50)
            )

            # Update axes
            fig.update_xaxes(
                showticklabels=False,  # Hide x-axis labels
                showline=True,
                linewidth=1,
                linecolor='rgba(128,128,128,0.2)'
            )
            
            fig.update_yaxes(
                gridcolor='rgba(128,128,128,0.2)',
                showgrid=True,
                zeroline=True,
                zerolinecolor='rgba(128,128,128,0.2)',
                zerolinewidth=1
            )

            # Return the figure instead of showing it
            return fig
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            return None