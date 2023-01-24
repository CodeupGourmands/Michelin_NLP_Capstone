def change_dtype_str(df):
    '''
    ## Description:
    This is a custom Function to change dtype to string
        as appropraiate for this project.
    ## Arguments:
    df = DataFrame
    ## Returns:
    df - DataFrame
    '''
    df.name = df.name.fillna('').astype('string')
    df.address = df.address.fillna('').astype('string')
    df.location = df.location.fillna('').astype('string')
    df.cuisine = df.cuisine.fillna('').astype('string')
    df.facilities_and_services = df.facilities_and_services.fillna(
        'NONE').astype('string')
    df.award = df.award.fillna('').astype('string')
    df.data = df.data.fillna('').astype('string')
    return df
