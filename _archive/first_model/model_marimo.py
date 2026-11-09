import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import os
    import marimo as mo
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import torch

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    return StandardScaler, mo, np, pd, sns, torch, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Prep
    """)
    return


@app.cell
def _(pd):
    """
    There are 14 attributes in each case of the dataset. They are:
    CRIM - per capita crime rate by town
    ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS - proportion of non-retail business acres per town.
    CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    NOX - nitric oxides concentration (parts per 10 million)
    RM - average number of rooms per dwelling
    AGE - proportion of owner-occupied units built prior to 1940
    DIS - weighted distances to five Boston employment centres
    RAD - index of accessibility to radial highways
    TAX - full-value property-tax rate per $10,000
    PTRATIO - pupil-teacher ratio by town
    B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT - % lower status of the population
    MEDV - Median value of owner-occupied homes in $1000's
    """
    # %%
    df = pd.read_csv("../data/BostonHousing.csv")
    df.describe()
    df.columns
    return (df,)


@app.cell
def _(df):
    # Data cleanup - remove NaN
    df.dropna(inplace=True)
    return


@app.cell
def _(df, pd):
    df_dummified = pd.get_dummies(df, drop_first=True, dtype=float, columns=["rad"])
    df_dummified
    return (df_dummified,)


@app.cell
def _(df_dummified):
    # Exploratory Data Analysis
    df_dummified.describe()
    return


@app.cell
def _(df, sns):
    # Correlation coefficients
    boston_corr = df.corr()
    sns.heatmap(boston_corr, annot=True, annot_kws={"size":8})
    return


@app.cell
def _(df_dummified):
    # seperate independent and dependent variables (target variable = y)
    X = df_dummified.drop(columns='medv')
    y = df_dummified['medv']
    f"X shape {X.shape} y shape {y.shape}"
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(StandardScaler, X_test, X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_train_scaled, scaler


@app.cell
def _(scaler):
    # check finished std deviations after scaling
    scaler.scale_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modelling
    """)
    return


@app.cell
def _(X_test_scaled, X_train_scaled, np, torch, y_test, y_train):
    # convert data to tensors for pytorch compatibility
    X_train_tensor = torch.from_numpy(X_train_scaled).float()
    X_test_tensor = torch.from_numpy(X_test_scaled).float()
    # single column vector to tensor
    y_train_tensor = torch.from_numpy(np.array(y_train)).reshape(-1,1).float()
    y_test_tensor = torch.from_numpy(np.array(y_test)).reshape(-1,1).float()
    return X_test_tensor, X_train_tensor, y_test_tensor, y_train_tensor


@app.cell
def _(torch):
    class LinRegTorch(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(LinRegTorch, self).__init__()
            self.linear = torch.nn.Linear(in_features=input_size, out_features=output_size)

        def forward(self, x):
            x = self.linear(x)
            return x

    return (LinRegTorch,)


@app.cell
def _(mo):
    learning_rate = mo.ui.slider(start=0.001, stop=0.01, step=0.001)
    learning_rate
    return (learning_rate,)


@app.cell
def _(LinRegTorch, X_train_scaled, learning_rate, torch):
    # model stores state internally somehow!
    model = LinRegTorch(
        input_size=X_train_scaled.shape[1],
        output_size=1
    )

    # optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate.value)
    loss_fn = torch.nn.MSELoss()
    return loss_fn, model, optimizer


@app.cell
def _(
    X_test_tensor,
    X_train_tensor,
    loss_fn,
    model,
    optimizer,
    torch,
    y_test_tensor,
    y_train_tensor,
):
    # Trainings-Loop (one big batch in multiple EPOCHS)
    EPOCHS = 5000

    losses_training, losses_test = [], []
    for epoch in range(EPOCHS):
        # forward pass -> run prediction
        y_train_pred = model(X_train_tensor)
        # calculate losses 
        loss = loss_fn(y_train_tensor, y_train_pred)
        # backward pass (calculate gradients from losses)
        loss.backward()
        # update model params
        optimizer.step()
        # zero out gradients
        optimizer.zero_grad()

        # save loss value for plotting
        losses_training.append(loss.item())
        #print(f"Epoch {epoch}: Loss {loss.item()}")

        # calculate losses against test data
        with torch.no_grad():
            y_test_pred = model(X_test_tensor)
            loss_test = loss_fn(y_test_tensor, y_test_pred)
            losses_test.append(loss_test.item())
    return losses_test, losses_training


@app.cell
def _(losses_test, losses_training, sns):
    sns.lineplot([losses_training, losses_test])
    return


if __name__ == "__main__":
    app.run()
