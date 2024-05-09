subroutine readdata()
    use global_variables
    ! reading input data
    open(2, file='input.dat', status='old')
    read(2,*) 
    read(2,*)
    read(2,*) restart, re_time
    read(2,*)
    read(2,*)
    read(2,*)
    read(2,*) nx, ny
    read(2,*)
    read(2,*)
    read(2,*) lx, ly
    read(2,*)
    read(2,*)
    read(2,*)
    read(2,*)
    read(2,*) w_ad, w_ppe, ad_itermax, ppe_itermax, solvetype_ad, solvetype_ppe
    read(2,*)
    read(2,*)
    read(2,*)
    read(2,*)
    read(2,*) errormax, tmax, dt, re, mu
    read(2,*)
    read(2,*)
    read(2,*)
    read(2,*) write_inter
    close(2)

    ! nx = nx + 2
    ! ny = ny + 2
    ! dx = lx/(nx) !!!!!!!!!!! to be commented later on
    ! dy = ly/(ny)
    nxf = nx + 1
    nyf = ny + 1
    nx = nx + 2
    ny = ny + 2
end subroutine

subroutine domain_init()
    use global_variables
    use files

    real :: dummy

    
    ! dxc = lx/(nx-2)
    ! dyc = ly/(ny-2)
    
    ! reading face coordinates
    open(x_face, file='xgrid.dat', status='old')
    do i = 1, nxf
        read(x_face,*) dummy, xf(i)
    enddo
    close(x_face)

    open(y_face, file='ygrid.dat', status='old')
    do i = 1, nyf
        read(y_face,*) dummy, yf(i)
    enddo
    close(y_face)

    ! setting up the cell centers
    do i = 2,nx-1
        x(i) = (xf(i-1)+xf(i))/2.0 
    end do
    
    do j = 2,ny-1
        y(j) = (yf(j-1)+yf(j))/2.0
    end do


    x(1) = -x(2)
    y(1) = -y(2)
    x(nx) = xf(nxf) + (xf(nxf) - x(nx-1)) 
    y(ny) = yf(nyf) + (yf(nyf) - y(ny-1))
    

    do j=2,ny-1
        do i=2,nx-1
            dx(i,j) = xf(i) - xf(i-1)
            dy(i,j) = yf(j) - yf(j-1)
        enddo
    enddo
    dx(1,:) = dx(2,:)
    dx(nx,:) = dx(nx-1,:)
    dx(:,1) = dx(:,2)
    dx(:,ny) = dx(:,ny-1)

    dy(1,:) = dy(2,:)
    dy(nx,:) = dy(nx-1,:)
    dy(:,1) = dy(:,2)
    dy(:,ny) = dy(:,ny-1)

    open(gridfile, file='grid.dat', status='unknown')
    write(*,*) 'writing grid data'
    write(gridfile,*) 'title = "post processing tecplot"'
    write(gridfile,*) 'variables = "x", "y", "dxi", "dyi"'
    write(gridfile,*) 'zone t="big zone", i=',nx,', j=',ny,', datapacking=point'

    do j=1,ny
        do i=1,nx
            write(gridfile,*) x(i),y(j),dx(i,j),dy(i,j)
        enddo
    enddo
    close(gridfile)

end subroutine

subroutine flow_init()

    use global_variables 
    use immersed_boundary
    use boundary_conditions

    character(len=50) :: fname, no, ext  
    real :: dump 
    
    if (restart .eq. 0) then
        u = 0
        v = 0
        uf = 0
        vf = 0
        p = 0
        t = 0
        call set_dirichlet_bc()
        call set_neumann_bc()

    else 
        t = re_time
        write_flag = int(t/dt)
        ! ext = '.dat'
        ! fname = 'Data/data.'
        ! write(no, "(i7.7)") re_time
        ! fname = trim(adjustl(fname))//no
        ! fname = trim(adjustl(fname))//trim(adjustl(ext))
        write(fname, "('Data/data.',I7.7,'.dat')") int(write_flag*dt)
        open(3, file=fname, status='unknown')
        read(3,*) 
        read(3,*) 
        read(3,*) 
        
        do j=1,ny
            do i = 1,nx
                read(3,*) x(i), y(j), u(i,j), v(i,j), p(i,j), dump, dump, dump
            end do
        end do
        close(3)
    end if

    call set_ssm_bc()
    
    do j=1,ny-2
        do i=2,nx-2
            uf(i,j) = iblank_fcu(i,j)*(u(i+1,j+1) + u(i,j+1))/2
        end do
    end do

    do j = 2,ny-2
        do i = 1,nx-2
            vf(i,j) = iblank_fcv(i,j)*(v(i+1,j+1) + v(i+1,j))/2
        end do 
    end do

    ! comment this out when appying top and bottom wall bc
    ! uf(nx,:) = iblank_fcu(nx,:)*(2*u(nx,2:ny-1) - uf(nx-1,:))
    ! vf(:,ny) = iblank_fcv(:,ny)*(2*v(2:nx-1,ny) - vf(:,ny-1))

end subroutine

subroutine probe_init()
    use global_variables
    use immersed_boundary
    use stats
    use files

    integer, dimension(1) :: xind, yind

    open(probe_input_file, file='probe.dat', status='old')
    read(probe_input_file,*) nprobes
    read(probe_input_file,*) 
    allocate(probe_loc(nprobes,2), probe_index_x(nprobes), probe_index_y(nprobes))
    do i = 1,nprobes
        read(probe_input_file,*) probe_loc(i,1), probe_loc(i,2)
    enddo
    close(probe_input_file)

    do i = 1,nprobes
        xind = minloc(abs(x(:)-probe_loc(i,1)))
        yind = minloc(abs(y(:)-probe_loc(i,2)))
        print *, xind, yind
        probe_index_x(i) = xind(1)
        probe_index_y(i) = yind(1)
    enddo

    open(probe_output_file, file='probe_output.dat', status='unknown')
    write(probe_output_file,*) nprobes, probe_loc, probe_index_x, probe_index_y
    write(probe_output_file,*) 
    write(probe_output_file,*) "Time  U   V  P"

endsubroutine 