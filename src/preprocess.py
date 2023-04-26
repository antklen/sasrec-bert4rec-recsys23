"""
Filter interactions.
"""


def add_time_idx(df, user_col='user_id', timestamp_col='timestamp', sort=True):
    """Add time index to interactions dataframe."""

    if sort:
        df = df.sort_values([user_col, timestamp_col])

    df['time_idx'] = df.groupby(user_col).cumcount()
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    return df


def filter_items(df, item_min_count, item_col='item_id'):

    print('Filtering items..')

    item_count = df.groupby(item_col).user_id.nunique()

    item_ids = item_count[item_count >= item_min_count].index
    print(f'Number of items before {len(item_count)}')
    print(f'Number of items after {len(item_ids)}')

    print(f'Interactions length before: {len(df)}')
    df = df[df.item_id.isin(item_ids)]
    print(f'Interactions length after: {len(df)}')

    return df


def filter_users(df, user_min_count, user_col='user_id'):

    print('Filtering users..')

    user_count = df.groupby(user_col).item_id.nunique()

    user_ids = user_count[user_count >= user_min_count].index
    print(f'Number of users before {len(user_count)}')
    print(f'Number of users after {len(user_ids)}')

    print(f'Interactions length before: {len(df)}')
    df = df[df.user_id.isin(user_ids)]
    print(f'Interactions length after: {len(df)}')

    return df


