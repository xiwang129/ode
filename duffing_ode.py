import jax.numpy as jnp 
import jax
from jax import random,grad,jit,vmap
from jax.experimental.ode import odeint
from matplotlib import pyplot as plt
import optax

def duffing_rhs(X,t):
    x,v = X  
    rhs = -alpha*x -beta*x**3 -delta *v + gamma*jnp.cos(omega*t)
    return jnp.array([v,rhs])

def solve_duffing(X0):
    # x0,v0 = X0
    # X0 =jnp.array([x0,v0]).astype('float32')
    t = jnp.arange(0,T,dt)
    solution = odeint(duffing_rhs,X0,t)
    return t,solution

def training_data_batch(batch_size):
    key = jax.random.PRNGKey(0)
    X0s = jax.random.uniform(key,(batch_size,2),minval=0,maxval=1)
    create_data_batch = vmap(solve_duffing,in_axes=(0))
    _,sol = create_data_batch(X0s)

    # extremely unsure this part; selected the last timesetp 
    x = sol[:,-1,0]
    y_target = sol[:,-1,1]
   
    # num_frequency = 50
    # n_samples = sol.shape[1] - num_frequency
    # x = jnp.stack([sol[:,i:i+num_frequency,:] for i in range(n_samples)])
    # y_target = jnp.stack([sol[:,i+num_frequency,:] for i in range(n_samples)])

    return x,y_target
 

def init_layer(key,n,m):
    k1,k2=random.split(key)
    w = random.normal(k1,(n,m))* jnp.sqrt(2.0/n)
    b = random.normal(k2,(m,)) * 0.1
    return w,b

def init_mlp(key,sizes): 
    keys = random.split(key,len(sizes)-1)
    return [init_layer(k,m,n) for k,m,n in zip(keys,sizes[:-1],sizes[:-1])]

def relu(x):
    return jnp.maximum(0,x)

def mlp(x,params):
    for w,b in params[:-1]:
        x = relu(jnp.dot(x,w)+b)
    w,b = params[-1]
    return jnp.dot(x,w)+b

def mse_loss(params,x,y):
    output_batch = vmap(mlp,in_axes=[0,None],out_axes=0)
    preds = output_batch(x,params)
    return jnp.mean((preds-y)**2)

# @jit
# def update(params,x,y,lr):
#     grads = grad(mse_loss)(params,x,y)
#     return [(w - lr*dw,b -lr*db) for (w,b),(dw,db) in zip(params,grads)]


def plot(test_inputs,trained_params):
    x = jnp.linspace(-2,2,20)
    y = jnp.linspace(-2,2,20)
    X,Y = jnp.meshgrid(x,y)
    _,sol = solve_duffing(test_inputs)
    U,V = sol[:,0],sol[:,1]
  
    # test_V = mlp(trained_params,U)
    plt.style.use('ggplot')
    fig,ax = plt.subplots(figsize=(10,8))
    ax.quiver(X,Y,U[:400],V[:400])
    # ax.quiver(X,Y,U[:400],test_V[:400])
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title('Duffing vector field')
    plt.show()


if __name__== "__main__":

    alpha,beta = -1.0,1.0
    delta,gamma,omega = 0.2,0.3,1.4
    dt = 0.01
    T = 100
    init_state = 0,0
 
    key = random.PRNGKey(0)
    input_dim = 32
    output_dim = 1
    lr = 1e-3
    batch_size = 32
    dim =64
    n_samples = 1000
    layer_sizes = [input_dim,dim,dim,output_dim]
    params = init_mlp(key,layer_sizes)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    losses = []

    for epoch in range(100):
        # params = update(params,X_train,y_train,lr)   
        X_train,y_target = training_data_batch(batch_size)
        loss,grads = mse_loss(params,X_train,y_target)
        updates,opt_state = optimizer.update(grads,opt_state)
        params = optax.apply_updates(params,updates)
        losses.append(loss)
    
    # plt.plot(losses)
    # plt.show()