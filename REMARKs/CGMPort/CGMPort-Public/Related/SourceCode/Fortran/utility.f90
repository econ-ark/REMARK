subroutine utility(values,ncols,gamma,util)
integer, INTENT(IN) :: ncols
REAL, INTENT(IN) :: values(1,ncols), gamma
REAL, INTENT(OUT) :: util(1,ncols)
  util = (values**(1-gamma))/(1-gamma)
end subroutine
