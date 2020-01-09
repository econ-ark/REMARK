subroutine spline(x,y,n,gam,y2)
implicit none
INTEGER, INTENT(IN) :: n
REAL, INTENT(IN) :: gam
REAL, INTENT(IN) :: x(n,1), y(n,1)
REAL, INTENT(OUT) :: y2(n,1)
INTEGER :: i, k
!REAL :: qn, un
REAL :: yp1
REAL :: p, sig
REAL, ALLOCATABLE :: u(:,:)
ALLOCATE(u(n,1))
yp1 = x(1,1)**(-gam)
y2(1,1)=-0.5
u(1,1)=(3.0/(x(2,1)-x(1,1)))*((y(2,1)-y(1,1))/(x(2,1)-x(1,1))-yp1)
do i=2,n-1
   sig = (x(i,1)-x(i-1,1))/(x(i+1,1)-x(i-1,1))
   p = sig*y2(i-1,1)+2.0
   y2(i,1) = (sig-1.0)/p
   u(i,1) = (6.0*((y(i+1,1)-y(i,1))/(x(i+1,1)-x(i,1))-(y(i,1)-y(i-1,1))/(x(i,1)-x(i-1,1)))/(x(i+1,1)-x(i-1,1))-sig*u(i-1,1))/p
end do
y2(n,1) =0.0
do k=n-1,1,-1
   y2(k,1) = y2(k,1)*y2(k+1,1)+u(k,1)
end do
DEALLOCATE(u)
RETURN
end subroutine

