SUBROUTINE  NTOI(value,nrow,grid,n,ind)
implicit none
INTEGER, INTENT(IN) :: nrow, n
REAL, INTENT(IN) :: value(nrow,1), grid(n,1)
INTEGER, INTENT(OUT) :: ind(nrow,1)
REAL, ALLOCATABLE :: ones(:,:), aux(:,:)
REAL :: step=0.0
ALLOCATE (aux(nrow,1))
aux = MIN(value,grid(n,1))
aux = MAX(aux,grid(1,1))
step = (grid(n,1)-grid(1,1))/(n-1)
ALLOCATE (ones(nrow,1))
ones = 1.0
ind = NINT(((aux-grid(1,1)*ones)/step)+ones)
DEALLOCATE(ones)
DEALLOCATE(aux)
RETURN
END subroutine



