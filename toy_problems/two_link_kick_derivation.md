# Two-step MPC with physical position kicks

This follows `open_closed_loop.pdf`: joint-space quadratic tracking, unit control
penalty, a two-step initial plan, then a one-step remaining-horizon feedback solve.
It is an analytic illustration of the disturbance mechanism; the notebooks retain
their nonlinear Cartesian objective, constraints, and differing horizons.

Write actual state as x (the user's disturbed state hat(q)), nominal predictions
as z, and exact measurements as y=x. With tau=dt,

$$x_{k+1}=x_k+\tau u_k+w_k,\qquad y_k=x_k,$$

where w_k is a position kick in radians, applied after command k. The nominal
optimizer assumes zero future kicks. There is no initial observation mismatch.

For references r1,r2, solve

$$\min_{u_0,u_1}\|x_0+\tau u_0-r_1\|^2+
\|x_0+\tau(u_0+u_1)-r_2\|^2+\|u_0\|^2+\|u_1\|^2.$$

Set d1=x0-r1, d2=x0-r2. The stationarity conditions, divided by two, are

$$\begin{bmatrix}1+2\tau^2&\tau^2\\\tau^2&1+\tau^2\end{bmatrix}
\begin{bmatrix}u_0^*\\u_1^*\end{bmatrix}
=-\tau\begin{bmatrix}d_1+d_2\\d_2\end{bmatrix}.$$

These equations apply independently to each joint. With D=1+3 tau^2+tau^4,

$$u_0^*=-\frac{\tau[(1+\tau^2)d_1+d_2]}D,\qquad
u_1^*=\frac{\tau[\tau^2d_1-(1+\tau^2)d_2]}D.$$

Open loop executes both fixed controls:

$$x_1=x_0+\tau u_0^*+w_0,\quad
x_2^{OL}=z_2+w_0+w_1,\quad
z_2=x_0-\frac{\tau^2[d_1+(\tau^2+2)d_2]}D.$$

Closed loop observes x1 exactly and solves the remaining one-step problem:

$$\min_v\|x_1+\tau v-r_2\|^2+\|v\|^2
\quad\Longrightarrow\quad
v^*=-\frac{\tau}{1+\tau^2}(x_1-r_2).$$

The original second command satisfies
u_1^*=-tau(z1-r2)/(1+tau^2), with z1=x0+tau u0*. Hence

$$v^*=u_1^*-\frac{\tau}{1+\tau^2}w_0,\qquad
x_2^{CL}=z_2+\frac{w_0}{1+\tau^2}+w_1.$$

For zero references, z2=x0/D, giving

$$\boxed{x_2^{OL}=x_0/D+w_0+w_1},\qquad
\boxed{x_2^{CL}=x_0/D+w_0/(1+\tau^2)+w_1}.$$

Feedback attenuates the first kick; the final kick arrives after the last command
and cannot be corrected within this two-step experiment. This does not imply
smaller realized error for every pair of kicks (they can cancel in open loop).
For independent zero-mean kicks with covariance Sigma, terminal disturbance
covariance is 2 Sigma in open loop and [1+(1+tau^2)^(-2)] Sigma in closed loop.

Over a longer open-loop execution with the same initial state,

$$x_k-z_k=\sum_{i=0}^{k-1}w_i.$$

In closed loop the nominal prediction is reset to the actual state each solve,
so x_(k+1)-z_(1|k)=w_k. Tracking error itself need not equal w_k: correction is
limited by the objective, command slew, geometry, and feasibility constraints.
The notebooks check both these identities using the executed kick schedule.
