#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <nccl.h>

static int is_server = 0;
static int n_proc = 0;
static int proc_id = 0;
static char server_ip[INET_ADDRSTRLEN];
static int server_port = 0;
static float values[1024]; // Max 1024 processes (128 nodes of 8x gpu)s
static ncclUniqueId nccl_id;
static int client_count = 0;
static int retry_limit = 5;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

typedef enum {
    REDUCE,
    BROADCAST
} OperationType;

void *handle_client(void *arg) {
    int client_socket = (intptr_t)arg;

    int proc_id_client;
    read(client_socket, &proc_id_client, sizeof(int));
    OperationType op_type;
    read(client_socket, &op_type, sizeof(OperationType));

    printf("\nClient process id: %d connected to server, Operation type: %d requested\n", proc_id_client, op_type);
    
    pthread_mutex_lock(&mutex);

    if (op_type == REDUCE) {
        float val;
        read(client_socket, &val, sizeof(float));
        values[client_count] = val; // store the client's value in the shared array
        client_count++;

        if (client_count == n_proc) { // if all processes (clients + server) have reported their values
            float sum = 0.0;
            for (int i = 0; i < n_proc; i++) {
                sum += values[i];
            }
            values[0] = sum; // store the result in values[0] for easy access
            client_count = 0; // reset client count for the next round
            pthread_cond_broadcast(&cond); // wake up all waiting clients
        } else {
            pthread_cond_wait(&cond, &mutex); // wait for the final result
        }
        write(client_socket, &values[0], sizeof(float)); // send the result back to the client
    } else if (op_type == BROADCAST) {
        ncclUniqueId dummy_id; // dummy variable, only for reading, will not be used
        read(client_socket, &dummy_id, sizeof(ncclUniqueId));
        client_count++;

        if (client_count == n_proc) { // if all processes have sent their values
            client_count = 0; // reset client count for the next round
            pthread_cond_broadcast(&cond); // wake up all waiting clients
        } else {
            pthread_cond_wait(&cond, &mutex); // wait for the broadcast value
        }
        write(client_socket, &nccl_id, sizeof(ncclUniqueId)); // send the result back to the client
    }

    pthread_mutex_unlock(&mutex);
    close(client_socket);
    return NULL;
}

void *server_function(void *arg) {
    int server_fd, client_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(server_port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, n_proc) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    printf("\nServer started\n");
    // wait forever accepting incoming client connections
    while (1) { 
        if ((client_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }
        // create a thread per incoming client connection
        pthread_t thread_id;
        pthread_create(&thread_id, NULL, handle_client, (void *)(intptr_t)client_socket);
        pthread_detach(thread_id);
    }
    return NULL;
}

void client_function(void *val, size_t data_size, OperationType op_type) {
    int sock = 0;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\nSocket creation error\n");
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(server_port);

    sleep(1);
    if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid server address, Client process: %d killed!\n", proc_id);
        exit(EXIT_FAILURE);
    }

    int n_retry = 0;
    while (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        if(++n_retry > 5)
        {
            printf("\nConnection failed to server, Client process: %d killed!\n", proc_id);
            exit(EXIT_FAILURE);
        }
        printf("\nConnection failed to server, Client process: %d retrying!\n", proc_id);
        sleep(1);
    }

    write(sock, &proc_id, sizeof(int)); // Send the process id
    write(sock, &op_type, sizeof(OperationType)); // Send the operation type
    write(sock, val, data_size); // Send the value
    read(sock, val, data_size); // recieve the value
    close(sock);
}

void distributed_init(int n_proc_param, int proc_id_param, const char *server_ip_param, int server_port_param) {
        
    n_proc = n_proc_param;
    proc_id = proc_id_param;
    strcpy(server_ip, server_ip_param);
    server_port = server_port_param;

    if (n_proc == 1) return;

    if (proc_id == 0) {
        is_server = 1;
        pthread_t server_thread;
        pthread_create(&server_thread, NULL, server_function, NULL);
    } else {
        is_server = 0;
    }
}

void distributed_reduce(float *val) {
    if (n_proc == 1) return;
        
    if (is_server) {
        pthread_mutex_lock(&mutex);
        values[client_count] = *val;
        client_count++; // store the server's value in the shared array
        if (client_count == n_proc) {
            float sum = 0.0;
            for (int i = 0; i < n_proc; i++) {
                sum += values[i];
            }
            values[0] = sum;
            client_count = 0;
            *val = sum;
            pthread_cond_broadcast(&cond);
        } else {
            pthread_cond_wait(&cond, &mutex);
            *val = values[0];
        }
        pthread_mutex_unlock(&mutex);
    } else {
        client_function(val, sizeof(float), REDUCE);
    }
}

void distributed_broadcast(ncclUniqueId *id) {
    if (n_proc == 1) return;

    if (is_server) {
        pthread_mutex_lock(&mutex);
        nccl_id = *id;
        client_count++;
        if (client_count == n_proc) {
            pthread_cond_broadcast(&cond);
        } else {
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);
    } else {
        client_function(id, sizeof(ncclUniqueId), BROADCAST);
    }
}

void distributed_barrier() {
    if (n_proc == 1) return;
    float dummy = 0.0;
    distributed_reduce(&dummy);
}