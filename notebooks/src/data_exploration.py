
def calculate_max_records_events(train_data, n=10):
    """
    Calculate the events with the maximum number of records.

    Parameters:
    train_data (pd.DataFrame): DataFrame containing 'event_id' column.
    n (int): Number of top events to return.

    Returns:
    pd.Index: Index of event_ids with the most records.
    """
    max_records_events = train_data.groupby('event_id').size().nlargest(n).index
    print(f"Events with most records: {max_records_events}")
    return max_records_events

def select_event_data(train_data, event_id):
    """
    Select data for a specific event.

    Parameters:
    train_data (pd.DataFrame): DataFrame containing 'event_id' column.
    event_id (int): The event_id to filter by.

    Returns:
    pd.DataFrame: DataFrame containing data for the specified event.
    """
    event_data = train_data[train_data['event_id'] == event_id]
    return event_data

def print_y_values_per_plane(top_event_data):
    """
    Print the mean Y_transformed values for each plane in the given event data.

    Parameters:
    top_event_data (pd.DataFrame): DataFrame containing 'plane' and 'Y_transformed' columns.
    """
    
    # validate input
    if 'plane' not in top_event_data.columns or 'Y_transformed' not in top_event_data.columns:
        raise ValueError("Input DataFrame must contain 'plane' and 'Y_transformed' columns.")
    
    # print mean values of y fro each palne
    for plane in sorted(top_event_data['plane'].unique()):
        plane_data = top_event_data[top_event_data['plane'] == plane]
        mean_y = plane_data['Y_transformed'].mean()
        print(f'Plane {plane}: Mean Y Transformed = {mean_y:.4f}')
    