subroutine evr(cash,nrow,ncol,v,nro,nco,fy,grid,secondd,ev_out)
INTEGER, INTENT(IN) :: nrow, ncol, nro, nco
REAL, INTENT(IN) :: cash(nrow,ncol), v(nro,nco), fy, grid(nco,1), secondd(nco,1)
REAL, INTENT(OUT) :: ev_out(nrow,ncol)
REAL, allocatable :: ones(:,:), inc(:,:), aux(:,:)
ALLOCATE (ones(nrow,ncol),inc(nrow,ncol),aux(nrow,ncol))
ev_out = 0.0
ones= 1.0
inc=fy*ones+cash
inc=MIN(inc,grid(nco,1))
inc=MAX(inc,grid(1,1))
!prob_li = 0.005
prob_li = 0.0
call splint(grid,v(1,:),secondd,nco,inc,nrow,ncol,aux)
ev_out=aux
!inc=2.0  !drop to 10%
inc=5.0  !drop to 25%
call splint(grid,v(1,:),secondd,nco,inc,nrow,ncol,aux)
ev_out=(1-prob_li)*ev_out + prob_li*aux
DEALLOCATE (ones,inc,aux)
end subroutine
