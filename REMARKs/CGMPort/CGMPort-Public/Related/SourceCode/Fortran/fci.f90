subroutine fci(sav,nrow,galfa,n,ret,rf,capinc)
implicit none
INTEGER, INTENT(IN) :: n, nrow
REAL, INTENT(IN) :: ret, rf
REAL, INTENT(IN) :: sav(nrow,1), galfa(n,1)
REAL, INTENT(OUT) :: capinc(nrow,n)
REAL, ALLOCATABLE :: rp(:,:), ones(:,:)
INTEGER :: ind1, ind2
ALLOCATE (rp(n,1), ones(n,1))
ones=1.0
do ind1=1,nrow
   rp = ret*galfa+rf*(ones-galfa)
   do ind2=1,n
      capinc(ind1,ind2) = sav(ind1,1)*rp(ind2,1)
   end do
end do
DEALLOCATE(rp,ones)
end subroutine


